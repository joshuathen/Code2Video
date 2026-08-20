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
            "High-dimensional geometry aids modern machine learning.",
            "The curse of dimensionality makes data points sparse.",
            "Finding patterns becomes harder in higher dimensions."
        ]
        self.setup_layout("Applications: Data Science and Beyond", lecture_lines)
        
        # Assets
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        satellite_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        high_dim_label = Text("High-Dim Space", font_size=24, color="#FF5733")
        self.place_at_grid(high_dim_label, "B3", scale_factor=0.7)
        self.place_at_grid(computer_icon, "A5", scale_factor=0.4)
        self.play(FadeIn(high_dim_label), FadeIn(computer_icon))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        point_cluster = VGroup(*[Dot(color="#33FF57", radius=0.05) for _ in range(50)])
        for p in point_cluster:
            p.move_to(self.grid["C3"] + np.random.uniform(-0.8, 0.8, 3))
        self.place_in_area(point_cluster, "B4", "D6", scale_factor=0.8)
        
        sparsity_label = Text("Sparsity", font_size=20, color="#33FF57")
        self.place_at_grid(sparsity_label, "B5")
        self.play(FadeIn(point_cluster), Write(sparsity_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3357FF")
        dr_label = Text("Dimension Reduction", font_size=24, color="#3357FF")
        self.place_in_area(dr_label, "D4", "E6", scale_factor=0.6)
        self.place_at_grid(satellite_icon, "E2", scale_factor=0.4)
        
        self.play(FadeIn(satellite_icon), Write(dr_label))
        self.wait(2)
