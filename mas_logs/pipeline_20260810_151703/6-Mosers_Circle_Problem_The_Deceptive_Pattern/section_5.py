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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion & Takeaway", [
            "Patterns need rigorous mathematical proof.",
            "Initial observations are only beginnings.",
            "Always stay skeptical of patterns."
        ])
        
        # Animations
        guess = MathTex(r"Guess: 2^{n-1}", color="#FF0000")
        truth = MathTex(r"Truth: \binom{n}{4} + \binom{n}{2} + 1", color="#00FF00")
        
        # 1. Show side-by-side
        # Updated per issue 38 / 31 / 32
        self.place_at_grid(guess, "B3", scale_factor=0.7)
        self.place_at_grid(truth, "C3", scale_factor=0.6)
        
        self.play(FadeIn(guess), FadeIn(truth))
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)
        
        # 2. Highlight polynomial for n=6
        highlight = SurroundingRectangle(truth, color="#00FF00", buff=0.1)
        self.play(Create(highlight))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)
        
        # 3. Detective icon
        detective = Circle(color="#FFFFFF").scale(0.3)
        # Updated per issue 38 / 33
        self.place_in_area(detective, "D5", "F6", scale_factor=0.9)
        self.play(FadeIn(detective))
        
        q_everything = Text("Question Everything!", font_size=30, color="#FFFFFF")
        self.place_at_grid(q_everything, "E3")
        self.play(FadeIn(q_everything))
        
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
