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
            "CLT bridges raw data and statistical inference.",
            "Ensure sample size is sufficiently large.",
            "Rule of thumb: use N over 30."
        ]
        self.setup_layout("Summary & Critical Takeaway", lecture_lines)
        
        # Setup visual elements
        raw_data_group = VGroup(
            *[Dot(point=np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0]), color=RED) for _ in range(50)]
        )
        norm_curve = Axes(x_range=[-3, 3], y_range=[0, 1]).plot(lambda x: np.exp(-x**2/2), color=BLUE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(raw_data_group, 'B1', 'D2', scale_factor=0.5)
        self.play(Create(raw_data_group), run_time=1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_in_area(norm_curve, 'C4', 'E6', scale_factor=0.5)
        self.play(FadeOut(raw_data_group), Create(norm_curve), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        n_label = Text("n > 30", font_size=36, color=GREEN)
        self.place_at_grid(n_label, 'B5', scale_factor=0.7)
        self.play(Write(n_label), run_time=1)
        self.wait(2)
