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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Intuitive Hook: The Flowing River", [
            "Vector fields assign direction to every point.",
            "Wind and water are natural examples.",
            "Points influence flow through expansion or rotation."
        ])
        
        # Assets
        river_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg", color="#2196F3")
        water_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg", color=WHITE)

        # Visuals
        stream = StreamLines(lambda p: np.array([0.5, 0.2*np.sin(p[0]), 0]), color="#2196F3")
        
        # Apply positioning constraints
        self.place_in_area(stream, "A3", "F6", scale_factor=0.8)
        self.place_at_grid(river_svg, "A2", scale_factor=0.5)
        
        velocity_label = Text("Flow Velocity", font_size=24, color=WHITE)
        self.place_at_grid(velocity_label, "F6", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(Create(stream), FadeIn(river_svg), run_time=2)
        self.lecture[0].set_color("#2196F3")

        # === Animation for Lecture Line 2 ===
        particles = VGroup(*[water_svg.copy().scale(0.05) for _ in range(5)])
        self.add(particles)
        self.play(FadeIn(particles), run_time=1)
        self.lecture[1].set_color("#4CAF50")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(velocity_label), run_time=1)
        self.lecture[2].set_color("#FFC107")
        self.wait(2)
