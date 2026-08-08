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
        self.setup_layout("Summary and Implications", ["Convergence depends on distance.", "Different rulers change our perspective.", "Math landscapes shift entirely."])
        
        # === Animation for Lecture Line 1 ===
        # Summarize 2-adic metric concept
        metric_txt = Text("2-adic Metric: |x|_2", font_size=30, color=WHITE)
        self.place_at_grid(metric_txt, 'A2', scale_factor=0.7)
        self.play(Write(metric_txt))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Flash key takeaway: 'Small norms imply closeness'
        takeaway = Text("Small norms imply closeness", font_size=28, color="#FFCC00")
        self.place_in_area(takeaway, 'D1', 'D3', scale_factor=0.8)
        self.play(FadeIn(takeaway))
        self.play(Indicate(takeaway))
        self.lecture[1].set_color("#FFCC00")

        # === Animation for Lecture Line 3 ===
        # Display concluding text: 'Convergence is context-dependent'
        conclusion = Text("Convergence is context-dependent", font_size=28, color=WHITE)
        self.place_in_area(conclusion, 'F1', 'F4', scale_factor=0.8)
        self.play(Create(conclusion))
        self.lecture[2].set_color(WHITE)
        
        self.wait(2)
