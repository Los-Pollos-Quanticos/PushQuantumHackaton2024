from manim import *
import numpy as np
import random
config.background_color = WHITE
DARK_TEXT = BLACK 

def Title(text, type="h1", color=DARK_TEXT):
        font_size = 48
        if type == "h1":
            font_size = 48
        elif type == "h2":
            font_size = 36
        elif type == "h3":
            font_size = 28

        return Text(
            text,
            font_size=font_size,
            color=color
        )

def Par(text, type="p2", color=DARK_TEXT):
    font_size = 30
    if type == "p1":
        font_size = 32
    elif type == "p2":
        font_size = 30
    elif type == "p3":
        font_size = 22

    return Text(
        text,
        font_size=font_size,
        color=color
    )

class Presentation(Scene):
    def gate(self, label, width=0.5, height=0.5, fill_color=BLUE,stroke_color=WHITE, text_color=BLACK):
        """
        Utility to create a rectangular "gate" with a centered label.
        """
        line = Line(color = BLACK)
        box = Rectangle(
            width=width,
            height=height,
            fill_color=fill_color,
            fill_opacity=1,
            stroke_color=stroke_color
        )
        gate_label = MathTex(label, color=text_color).scale(0.6)
        gate_group = VGroup(box, gate_label)
        gate_label.move_to(box.get_center())
        return gate_group
    
    def place_gate(self, gate_obj, line_obj, shift_x=0, shift_y=0):
        """
        Move a gate to the center of a line (Rectangle of height=0)
        then shift it horizontally or vertically.
        """
        center = line_obj.get_center()
        gate_obj.move_to([center[0] + shift_x, center[1] + shift_y, 0])
        return gate_obj
    
    def rodeo_circuit(self, step):
        # Build circuit elements
        a1 = MathTex(r"|0>", color=BLACK).to_edge(UL + DOWN * 0.1)
        a2 = MathTex(r"|0>", color=BLACK).next_to(a1, DOWN, buff=0.4)
        a3 = MathTex(r"|0>", color=BLACK).next_to(a2, DOWN, buff=1.5)
        init = MathTex(r"|\psi_I>", color=BLACK).next_to(a3, DOWN, buff=1)
        
        # Create main horizontal lines for each state label
        lines_main = VGroup()
        for obj in (a1, a2, a3, init):
            line = Rectangle(color=BLACK, width=11.5, height=0)
            line.next_to(obj)
            lines_main.add(line)
        
        # Additional horizontal lines above and below the last line
        lu1 = Rectangle(color=BLACK, width=11.5, height=0)
        lu2 = Rectangle(color=BLACK, width=11.5, height=0)
        ld1 = Rectangle(color=BLACK, width=11.5, height=0)
        ld2 = Rectangle(color=BLACK, width=11.5, height=0)
        li = lines_main[-1]
        
        lu1.next_to(li, UP, buff=0.2)
        lu2.next_to(lu1, UP, buff=0.2)
        ld1.next_to(li, DOWN, buff=0.2)
        ld2.next_to(ld1, DOWN, buff=0.2)
        
        # Group all line elements together
        lines_all = VGroup(a1, a2, a3, init, lines_main, lu1, lu2, ld1, ld2)
        
        if step==1:
            return lines_all
        
        # Left H gates
        H_left = VGroup()
        for line_obj in lines_main[:3]:
            H_gate = self.gate("H", stroke_color=BLUE)
            self.place_gate(H_gate, line_obj, shift_x=-5)
            H_left.add(H_gate)
        
        if step==2:
            return VGroup(lines_all, H_left)
        
        # Exponential evolution gates and control lines
        e_labels = [r"e^{-iH_{obj}t_{1}}", r"e^{-iH_{obj}t_{2}}", r"e^{-iH_{obj}t_{3}}"]
        x_shifts = [-4.1, -1.9, 1.5]
        y_shifts = [3.9, 3, 1]
        e_gates = VGroup()
        controls = VGroup()
        for i, lbl in enumerate(e_labels):
            gate_obj = self.gate(lbl, width=1.5, height=1, stroke_color=BLACK, fill_color=WHITE)
            self.place_gate(gate_obj, li, shift_x=x_shifts[i])
            e_gates.add(gate_obj)
            
            ctrl_line = Rectangle(color=BLACK, width=0, height=y_shifts[i])
            ctrl_line.next_to(gate_obj, UP, buff=0)
            controls.add(ctrl_line)
            
            circle = Circle(radius=0.05, color=BLACK, fill_opacity=1)
            circle.move_to([gate_obj.get_center()[0], lines_main[i].get_center()[1]-0.06, 0])
            controls.add(circle)

        if step==3:
            return VGroup(lines_all, H_left, e_gates, controls)
        
        p_labels = [r"P(Et_{1})", r"P(Et_{2})", r"P(Et_{3})"]
        p_gates = VGroup()
        for i, lbl in enumerate(p_labels):
            p_gate = self.gate(lbl, width=1.3, height=0.8, stroke_color=RED_C, fill_color=RED_C)
            circle_center = controls[2*i + 1].get_center()
            p_gate.move_to([circle_center[0] + 1, circle_center[1], 0])
            p_gates.add(p_gate)

        if step==4:
            return VGroup(lines_all, H_left, e_gates, controls, p_gates)
        
        # Right H and measurement gates
        H_right = VGroup()
        M_gates = VGroup()
        for line_obj in lines_main[:3]:
            H_gate = self.gate("H", stroke_color=BLUE)
            self.place_gate(H_gate, line_obj, shift_x=4.2)
            H_right.add(H_gate)
            
            M_gate = self.gate("M", stroke_color=BLACK, fill_color=BLACK, text_color=WHITE)
            self.place_gate(M_gate, line_obj, shift_x=5)
            M_gates.add(M_gate)

        if step==5:
            return VGroup(lines_all, H_left, e_gates, controls, p_gates, H_right, M_gates)
        
        circuit = VGroup(lines_all, H_left, e_gates, controls, p_gates, H_right, M_gates)
        return circuit

    def construct(self):
        self.Head()
        self.next_slide()

        self.Outline()
        self.next_slide()

        self.Intro()
        self.next_slide()

        self.Rodeo1()
        self.next_slide()

        self.Rodeo2()
        self.next_slide()

        self.Rodeo3()
        self.next_slide()

        self.wait(10)

    def next_slide(self):
        self.wait(1)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def Head(self):
        h1 = Title(
            "Estimation of molecular spectra using",
        ).move_to(ORIGIN+UP*0.8)
        h1s = Title(
            "the Rodeo Algorithm",
        ).next_to(h1, DOWN, buff=0.2)

        h3 = Title(
            "Presented by",
            "h3",
        )
        h3.next_to(h1s, DOWN, buff=0.5)
        presenters = Title(
            "Zanasi Francesco, Parentin Marco, Straccali Leonardo, Muscarà Fedrerico",
            "h3",
        )
        presenters.next_to(h3, DOWN)
        
        self.play(FadeIn(h1,h1s,h3,presenters))

    def Outline(self):
        h1 = Title(
            "Outline",
        ).to_edge(UL)

        h2 = Title(
            "Rodeo Algorithm for Quantum Computing",
            "h2",
        ).to_edge(UL).shift(DOWN)

        h3 = Title(
            "Kenneth Choi, Dean Lee, Joey Bonitati, Zhengrong Qian and Jacob Watkins \nhttps://doi.org/10.1103/PhysRevLett.127.040505",
            "h3"
        ).to_edge(UL).shift(DOWN*1.5)

        image_size = 2.7
        image1 = ImageMobject("./images/mechanism.png").set_height(image_size)
        image2 = ImageMobject("./images/h2.png").set_height(image_size)
        image3 = ImageMobject("./images/h2o.png").set_height(image_size)

        text1 = Par("Algorithm \nStructure", "p2")
        text2 = Par("Test full spectrum \nH2 molecule", "p2")
        text3 = Par("Test full spectrum \nH2O molecule", "p2")

        block1 = Group(image1, text1).arrange(DOWN, buff=0.2)
        block2 = Group(image2, text2).arrange(DOWN, buff=0.2)
        block3 = Group(image3, text3).arrange(DOWN, buff=0.2)

        blocks = Group(block1, block2, block3).arrange(RIGHT, buff=1)
        blocks.move_to(ORIGIN).shift(DOWN * 1)

        self.play(FadeIn(h1, h2, h3, blocks))

    def Intro(self):
        h1 = Title(
            "Introduction to the field",
        ).to_edge(UL)

        self.play(FadeIn(h1))

    def Rodeo1(self):
        h1 = Title(
            "Rodeo Algorithm",
        ).to_edge(UL)

        img = ImageMobject("./images/rodeo_logo.png").set_height(3)

        par = Par(
            "The Rodeo Algorithm is a quantum algorithm that estimates the eigenvalues of a Hamiltonian operator."
        ).next_to(h1, DOWN, buff=0.5)

        self.play(FadeIn(h1, img))

    def Rodeo2(self):
        title = Text("Rodeo Structure", font_size=48, color=BLACK).to_edge(UL)
        self.play(FadeIn(title))

        circuit = self.rodeo_circuit(1).move_to(ORIGIN).scale(0.8)
        expr = MathTex(
            r"\lvert 0 \rangle \otimes \lvert \psi_{I} \rangle",
            color=BLACK
        ).to_edge(DOWN)
        self.play(Write(circuit), Write(expr))
        self.wait(1)

        circuit = self.rodeo_circuit(2).move_to(ORIGIN).scale(0.8)
        tra = MathTex(
            r"\frac{1}{\sqrt{2}}(\lvert 0 \rangle + \lvert 1 \rangle) \otimes \lvert \psi_{I} \rangle",
            color=BLACK
        ).to_edge(DOWN)
        self.play(Write(circuit[1]), Transform(expr, tra))
        self.wait(1)
        
        circuit = self.rodeo_circuit(3).move_to(ORIGIN).scale(0.8)
        tra = MathTex(
            r"\frac{1}{\sqrt{2}}(\lvert 0 \rangle \lvert \psi_{I} \rangle + e^{-iHt_{n}} \lvert 1 \rangle \lvert \psi_{I} \rangle)",
            color=BLACK
        ).to_edge(DOWN)
        self.play(Write(circuit[3]), Write(circuit[2]), Transform(expr, tra))
        self.wait(1)

        circuit = self.rodeo_circuit(4).move_to(ORIGIN).scale(0.8)
        tra = MathTex(
            r"\frac{1}{\sqrt{2}}(\lvert 0 \rangle \lvert \psi_{I} \rangle + e^{-i(H-E)t_{n}} \lvert 1 \rangle \lvert \psi_{I} \rangle)",
            color=BLACK
        ).to_edge(DOWN)
        self.play(Write(circuit[4]), Transform(expr, tra))
        self.wait(1)

        circuit = self.rodeo_circuit(5).move_to(ORIGIN).scale(0.8)
        tra = MathTex(
            r"\frac{1}{2}[(1+e^{-i(H-E)t_{n}})\lvert 0 \rangle \lvert \psi_{I} \rangle + (1-e^{-i(H-E)t_{n}})\lvert 1 \rangle \lvert \psi_{I} \rangle)]",
            color=BLACK
        ).to_edge(DOWN)
        self.play(Write(circuit[5]), Transform(expr, tra))

    def Rodeo3(self):
        title = Text("Measurement probability", font_size=48, color=BLACK).to_edge(UL)
        
        expr = MathTex(
            r"\frac{1}{2}[(1+e^{-i(H-E)t_{n}})\lvert 0 \rangle \lvert \psi_{I} \rangle + (1-e^{-i(H-E)t_{n}})\lvert 1 \rangle \lvert \psi_{I} \rangle)]",
            color=BLACK
        ).move_to(ORIGIN)
        self.play(FadeIn(title), Write(expr, run_time=1))
        self.wait(1)

        tra = MathTex(
            r"P^{\lvert0\rangle}(E_{obj},E,t_{n})=\lvert\frac{1}{2}+\frac{1}{2}e^{-i(E_{obj}-E)t_{n}}\lvert^2",
            color=BLACK
        ).move_to(ORIGIN)
        self.play(Transform(expr, tra))
        self.wait(1)

        tra = MathTex(
            r"P^{\lvert0\rangle}(E_{obj},E,t_{n})=\cos^2\left[(E_{obj}-E)\frac{t_n}{2}\right]\\",
            color=BLACK
        ).move_to(ORIGIN)
        self.play(Transform(expr, tra))
        self.wait(1)

        tra = MathTex(
            r"P^{\lvert0\rangle}_N(E_{obj},E) =\prod_{n=1}^{N} \cos^2\left[(E_{obj}-E)\frac{t_n}{2}\right]\\",
            color=BLACK
        ).move_to(ORIGIN)
        self.play(Transform(expr, tra))

        self.play(expr.animate.to_edge(UL).shift(DOWN*0.7+LEFT).scale(0.8), run_time=1)
        self.wait(1)

        N_max = 8
        T_RMS = 10.0
        t_n = [11,13,5,12,11,8,10,12]

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.1, 0.2],
            tips=False,
            axis_config={"color": BLACK, "stroke_width": 2},
        ).add_coordinates().shift(DOWN * 1)
        axes.scale(0.8)

        x_label = MathTex(r"E_{\text{obj}} - E", color=BLACK).next_to(axes.x_axis, UR, buff=0.2).shift(LEFT * 0.5)
        y_label = MathTex(r"P_N", color=BLACK).next_to(axes.y_axis, UR, buff=0.2).shift(DOWN * 0.5)
        x_label.scale(0.8)
        y_label.scale(0.8)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.wait(1)

        def P_N_func(x, N):
            prod_val = 1.0
            for i in range(N):
                prod_val *= np.cos(0.5 * x * t_n[i])**2
            return prod_val

        def get_plot_for_N(N, color=BLUE):
            xs = np.linspace(-3, 3, 1000)
            points = [axes.c2p(x, np.abs(P_N_func(x, N))) for x in xs]
            plot_obj = VMobject()
            plot_obj.set_points_smoothly(points)
            plot_obj.set_color(color)
            return plot_obj

        param_label = Tex(f"N = 1, $T_\\mathrm{{RMS}} = {T_RMS}$", font_size=36, color=BLACK)
        param_label.next_to(axes.y_axis, UR).shift(RIGHT * 2)
        self.add(param_label)

        plot_old = get_plot_for_N(1)
        self.play(Create(plot_old))
        self.wait(1)

        for N in range(2, N_max + 1):
            plot_new = get_plot_for_N(N)
            new_label = Tex(f"N = {N}, $T_\\mathrm{{RMS}} = {T_RMS}$", font_size=36, color=BLACK)
            new_label.next_to(axes.y_axis, UR).shift(RIGHT * 2)

            self.play(
                Transform(plot_old, plot_new),
                Transform(param_label, new_label),
                run_time=1
            )
            self.wait(0.5)
