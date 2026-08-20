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
        self.setup_layout("Application: Why Does This Matter?", [
            "The CLT bridges samples and populations.",
            "It powers inferential statistics and quality.",
            "Averages provide reliable predictability."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display real-world data like coin flips: #F1C40F
        # Use [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg]
        coin_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg"
        data_viz = VGroup(*[SVGMobject(coin_path, color="#F1C40F") for _ in range(5)])
        data_viz.arrange(RIGHT, buff=0.2)
        self.place_at_grid(data_viz, 'B4', scale_factor=0.7)
        self.play(Create(data_viz), run_time=1.5)
        self.lecture[0].set_color("#F1C40F")

        # === Animation for Lecture Line 2 ===
        # Show how CLT predicts outcomes for large groups: #3498DB
        bell_curve = FunctionGraph(lambda x: np.exp(-x**2 / 2), x_range=[-3, 3], color="#3498DB")
        self.place_in_area(bell_curve, 'D3', 'F5', scale_factor=0.6)
        self.play(Create(bell_curve), run_time=1.5)
        self.lecture[1].set_color("#3498DB")

        # === Animation for Lecture Line 3 ===
        # Contrast unpredictable individual coin flips [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg] with predictable aggregate: #E74C3C.
        individual = SVGMobject(coin_path, color="#E74C3C")
        self.place_at_grid(individual, 'B5', scale_factor=0.8)
        self.play(FadeIn(individual), run_time=1)
        self.lecture[2].set_color("#E74C3C")
        self.wait(2)
