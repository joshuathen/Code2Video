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
        self.setup_layout("The Math: Visualizing Attention Weights", [
            "Attention matches queries to keys.", 
            "Higher scores indicate stronger relevance.", 
            "Softmax normalizes these scores to weights."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display matrix multiplication WQ * X
        eq = MathTex(r"W_Q X", color=WHITE)
        self.place_at_grid(eq, "B2", scale_factor=0.8)
        self.play(Write(eq))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight dot product result in #FFD700 color
        dot_product = MathTex(r"Q K^T", color="#FFD700")
        self.place_at_grid(dot_product, "D2", scale_factor=0.8)
        self.play(FadeIn(dot_product), self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        # Show Softmax normalization curve in #00FFFF color
        softmax = MathTex(r"\text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)", color="#00FFFF")
        self.place_at_grid(softmax, "F2", scale_factor=0.7)
        self.play(FadeIn(softmax), self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
