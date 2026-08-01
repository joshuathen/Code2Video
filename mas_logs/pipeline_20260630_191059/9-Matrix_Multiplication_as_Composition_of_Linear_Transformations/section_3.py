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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize title and lecture lines
        self.setup_layout("Function Composition Logic (The 'Right-to-Left' Rule)", [
            "- Transformations work like nested functions.",
            "- We apply A first, then B.",
            "- The first transformation matrix goes on the right."
        ])

        # === Animation for Lecture Line 1 ===
        # Display 'f(g(x))' and 'B(A(v))' with 'g' and 'A' in yellow (#FFFF00).
        # Replaced MathTex with VGroup of Text to avoid 'latex' dependency
        func_comp = VGroup(Text("f("), Text("g"), Text("(x))")).arrange(RIGHT, buff=0.05)
        func_comp[1].set_color("#FFFF00")
        
        mat_comp = VGroup(Text("B("), Text("A"), Text("(v))")).arrange(RIGHT, buff=0.05)
        mat_comp[1].set_color("#FFFF00")
        
        self.place_at_grid(func_comp, 'B2', scale_factor=1.2)
        # Issue 33: Move mat_comp to B4 to reduce horizontal gap
        self.place_at_grid(mat_comp, 'B4', scale_factor=1.2)
        
        # Highlight first lecture line
        self.lecture[0].set_color("#FFFF00")
        self.play(Write(func_comp), Write(mat_comp))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the equation Vector_final = B * A * Vector_initial in white (#FFFFFF).
        # Replaced MathTex with MarkupText/Text to avoid 'latex' dependency
        eq_v_final = MarkupText("v<sub>final</sub>")
        eq_equal = Text("=")
        eq_b = Text("B")
        eq_dot1 = Text("·")
        eq_a = Text("A")
        eq_dot2 = Text("·")
        eq_v_initial = MarkupText("v<sub>initial</sub>")
        
        equation = VGroup(eq_v_final, eq_equal, eq_b, eq_dot1, eq_a, eq_dot2, eq_v_initial).arrange(RIGHT, buff=0.15)
        # Issue 32: Move equation to area D2-D6 and adjust scale to prevent crowding
        self.place_in_area(equation, 'D2', 'D6', scale_factor=1.0)
        
        # Transition highlight to second lecture line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        self.play(Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate a glow moving from Vector_initial to matrix A, then to matrix B.
        # Transition highlight to third lecture line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # Creating a glow effect using a Highlight Rectangle
        glow = SurroundingRectangle(eq_v_initial, color="#FFFF00", buff=0.1)
        glow.set_stroke(width=4, opacity=0.8)
        
        self.play(Create(glow))
        self.play(glow.animate.move_to(eq_a), run_time=1.5)
        self.play(Indicate(eq_a, color="#FFFF00"))
        self.play(glow.animate.move_to(eq_b), run_time=1.5)
        self.play(Indicate(eq_b, color="#FFFF00"))
        self.play(FadeOut(glow))
        self.wait(2)
