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
        self.setup_layout("Prerequisite: The Exponential Foundation", 
                          ["The base is our starting unit.", 
                           "The exponent is our growth intensity.", 
                           "The result is our final total."])
        
        # Main expression
        expr = MathTex("b", "^", "{x}", "=", "y", font_size=48)
        self.place_in_area(expr, 'B3', 'C4', scale_factor=1.2)
        self.add(expr)

        # Labels/Annotations
        formula_label = Text("Exponential Form", font_size=20)
        self.place_at_grid(formula_label, 'C5', scale_factor=0.9)
        self.add(formula_label)

        secondary_annotation = Text("Growth Mechanism", font_size=18, color=BLUE)
        self.place_in_area(secondary_annotation, 'E3', 'F4', scale_factor=0.7)
        self.add(secondary_annotation)

        # Assets (Using placeholders since files are likely empty/not usable icons as per path)
        # Note: The provided paths are not real image files, so we skip adding them to avoid errors.

        # === Animation for Lecture Line 1 ===
        # Line: "The base is our starting unit."
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Indicate(expr[0]))

        # === Animation for Lecture Line 2 ===
        # Line: "The exponent is our growth intensity."
        self.play(self.lecture[1].animate.set_color("#FF6600"))
        self.play(expr[0].animate.set_color("#FF6600"), Indicate(expr[0]))

        # === Animation for Lecture Line 3 ===
        # Line: "The result is our final total."
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(expr[2].animate.set_color("#FFFF00"), Indicate(expr[2]))
