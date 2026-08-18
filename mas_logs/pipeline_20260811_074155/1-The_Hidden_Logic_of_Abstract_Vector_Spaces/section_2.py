from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "An abstract vector space follows specific mathematical rules.",
            "These rules apply to sets of any mathematical objects.",
            "The \"Vector Club\" requires four rules for vector addition.",
            "It also demands four rules for scaling by numbers.",
            "If these axioms hold, the objects are official vectors."
        ]
        self.setup_layout("The Rulebook: 8 Axioms of the Vector Club", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Use Asset: gate.svg
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        gate = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gate.svg").set_color("#FFD700")
        self.place_in_area(gate, "C3", "E5", scale_factor=1.5)
        
        gate_label = Text("VECTOR CLUB", font_size=20, color="#FFD700")
        self.place_at_grid(gate_label, "B4")
        
        symbol_v = MathTex("V", color=WHITE)
        symbol_f = MathTex("F", color=WHITE)
        self.place_at_grid(symbol_v, "D3", scale_factor=1.2)
        self.place_at_grid(symbol_f, "D5", scale_factor=1.2)
        
        self.play(FadeIn(gate), Write(gate_label), Write(symbol_v), Write(symbol_f))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF69B4")
        )
        
        # Pink Matrix - Fix Issue 21: Positioned at F3
        matrix_box = Square(side_length=0.8, color="#FF69B4")
        matrix_content = MathTex(r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}", font_size=18, color="#FF69B4")
        matrix_obj = VGroup(matrix_box, matrix_content)
        self.place_at_grid(matrix_obj, "F3")
        
        # Blue Function - Fix Issue 22: Positioned at F5
        func_box = Square(side_length=0.8, color="#1E90FF")
        func_content = MathTex("f(x) = \sin(x)", font_size=18, color="#1E90FF")
        func_obj = VGroup(func_box, func_content)
        self.place_at_grid(func_obj, "F5")
        
        self.play(FadeIn(matrix_obj), FadeIn(func_obj))
        # Objects approach the gate
        self.play(
            matrix_obj.animate.move_to(self.grid["E3"]),
            func_obj.animate.move_to(self.grid["E5"]),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00")
        )
        
        addition_title = Text("Addition Axioms", font_size=20, color="#00FF00")
        self.place_at_grid(addition_title, "A4")
        
        checkmarks_add = VGroup(*[
            MathTex(r"\checkmark", color="#00FF00", font_size=24) for _ in range(4)
        ])
        # Arrange checkmarks in a small grid above/around the gate
        for i, check in enumerate(checkmarks_add):
            self.place_at_grid(check, f"B{i+2}")

        self.play(Write(addition_title))
        self.play(LaggedStart(*[FadeIn(c) for c in checkmarks_add], lag_ratio=0.3))
        
        # Objects move closer to gate
        self.play(
            matrix_obj.animate.move_to(self.grid["D3"]),
            func_obj.animate.move_to(self.grid["D5"]),
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#00FF00")
        )
        
        # Fix Issue 20: Positioned at D4 to avoid overlap with gate top
        scaling_title = Text("Scaling Axioms", font_size=20, color="#00FF00")
        self.place_at_grid(scaling_title, "D4")
        
        checkmarks_scale = VGroup(*[
            MathTex(r"\checkmark", color="#00FF00", font_size=24) for _ in range(4)
        ])
        # Position checkmarks in row C
        for i, check in enumerate(checkmarks_scale):
            self.place_at_grid(check, f"C{i+2}")

        self.play(Write(scaling_title))
        self.play(LaggedStart(*[FadeIn(c) for c in checkmarks_scale], lag_ratio=0.3))
        
        # Objects move to entrance center
        self.play(
            matrix_obj.animate.move_to(self.grid["C4"]),
            func_obj.animate.move_to(self.grid["C4"]),
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFFFFF")
        )
        
        # Gate opens: scale and fade out to simulate opening/passing
        self.play(
            gate.animate.scale(2).set_opacity(0),
            FadeOut(gate_label),
            FadeOut(symbol_v),
            FadeOut(symbol_f),
            FadeOut(addition_title),
            FadeOut(scaling_title),
            FadeOut(checkmarks_add),
            FadeOut(checkmarks_scale)
        )
        
        # Move objects to center of Vector Space (D4) and glow
        self.play(
            matrix_obj.animate.move_to(self.grid["D4"]).set_color(WHITE),
            func_obj.animate.move_to(self.grid["D4"]).set_color(WHITE),
            run_time=1
        )
        
        glow = Circle(radius=1.2, color=WHITE, fill_opacity=0.3).move_to(self.grid["D4"])
        official_label = Text("OFFICIAL VECTORS", font_size=28, color=WHITE)
        self.place_at_grid(official_label, "A4")
        
        self.play(
            FadeIn(glow, scale=1.2),
            Write(official_label),
            matrix_obj.animate.scale(1.2),
            func_obj.animate.scale(1.2)
        )
        self.wait(2)

        # Final Cleanup
        self.play(
            FadeOut(matrix_obj),
            FadeOut(func_obj),
            FadeOut(glow),
            FadeOut(official_label)
        )
