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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Why do we multiply rows by columns?", "The product columns track the basis vectors.", "They show where i and j land."]
        self.setup_layout("The Calculation Rule", lecture_lines)
        
        # Define objects
        eqn = MathTex(r"A \cdot (B \cdot \vec{x}) = \vec{y}").set_color(WHITE)
        
        # Apply positioning fix for issue 30
        self.place_at_grid(eqn, 'C3', scale_factor=1.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(eqn))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFCC00"))
        # Highlight B * x part
        highlight_box = SurroundingRectangle(eqn[0][2:7], color="#FFCC00", buff=0.1)
        # Apply positioning fix for issue 29
        self.place_in_area(highlight_box, 'B3', 'D6', scale_factor=0.9)
        self.play(Create(highlight_box))
        self.wait(1)
        self.play(FadeOut(highlight_box))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFCC"))
        # Highlight A * (B * x) part
        highlight_box2 = SurroundingRectangle(eqn[0][0:9], color="#00FFCC", buff=0.1)
        self.play(Create(highlight_box2))
        self.wait(2)
