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
        lecture_lines = [
            "Simple rules create complex population behavior.",
            "Models help predict herd immunity thresholds.",
            "Further study can include more advanced models."
        ]
        self.setup_layout("Conclusion & Real-world Application", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Simple rules create complex population behavior.
        title_app = Text("Real World Applications", color="#FFFFFF")
        # Fixed positioning per issue 30
        self.place_at_grid(title_app, 'A3', scale_factor=0.6)
        self.play(Write(title_app))
        self.lecture[0].set_color("#FFD700") # Gold

        # === Animation for Lecture Line 2 ===
        # Models help predict herd immunity thresholds.
        herd_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/herd.svg")
        outbreak_list = VGroup(
            Text("Herd immunity", color="#ADD8E6"),
            Text("Outbreak prediction", color="#ADD8E6")
        ).arrange(DOWN, aligned_edge=LEFT)
        
        examples = VGroup(herd_icon, outbreak_list).arrange(RIGHT, buff=0.3)
        # Fixed positioning per issue 31
        self.place_in_area(examples, 'B4', 'D6', scale_factor=0.5)
        self.play(FadeIn(examples))
        self.lecture[1].set_color("#00CED1") # DarkTurquoise

        # === Animation for Lecture Line 3 ===
        # Further study can include more advanced models.
        summary = Text("Model informs decisions", color="#FFFF00", font_size=20)
        # Fixed positioning per issue 32
        self.place_at_grid(summary, 'E4', scale_factor=0.7)
        self.play(Indicate(summary))
        self.lecture[2].set_color("#FF6347") # Tomato
        
        self.wait(2)
