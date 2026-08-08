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
        lecture_lines = ["Roots are partial growth cycles.", "Cube root of twenty-seven is three.", "Fractional exponents represent these roots."]
        self.setup_layout("Roots as Fractional Exponents", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Roots are partial growth cycles.
        # Display radical symbol: n√x (Color: #FFFF00)
        radical = MathTex(r"\sqrt[n]{x}", color="#FFFF00")
        self.place_at_grid(radical, 'C2', scale_factor=1.2)
        self.play(Write(radical))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Cube root of twenty-seven is three.
        # Transform n√x into x^(1/n) smoothly (Color: #00FF00)
        cube_root_27 = MathTex(r"\sqrt[3]{27} = 3", color="#00FF00")
        self.place_at_grid(cube_root_27, 'E2', scale_factor=1.0)
        
        self.play(FadeOut(radical))
        self.play(Write(cube_root_27))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fractional exponents represent these roots.
        # Highlight the fraction 1/n to emphasize exponent behavior.
        fractional_exp = MathTex(r"x^{1/n}", color="#FF00FF")
        self.place_at_grid(fractional_exp, 'C4', scale_factor=1.2)
        
        self.play(FadeOut(cube_root_27))
        self.play(Write(fractional_exp))
        
        # Highlight fraction (subscript of exponent)
        # fractional_exp[0] is the MathTex object's Mobject list
        # Using a safer way to get the sub-part
        fraction = fractional_exp[0][2:]
        self.play(Indicate(fraction, color="#FFFF00"))
        
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
