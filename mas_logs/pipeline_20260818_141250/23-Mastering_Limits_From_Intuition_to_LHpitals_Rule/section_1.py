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
        self.setup_layout("Intuitive Foundation & The Epsilon-Delta Definition", [
            "Limits represent approaching a target value.",
            "A squirrel approaches a nut at L.",
            "Epsilon defines the tolerance around the limit.",
            "Delta defines proximity to the target x.",
            "We trap the squirrel within the tolerance."
        ])
        
        # Axes and Function
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True})
        func = lambda x: 0.5 * (x - 2)**2 + 2
        graph = axes.plot(func, color=WHITE)
        
        right_side = VGroup(axes, graph)
        # Using A3-F5 as requested for grid constraints
        self.place_in_area(right_side, 'A3', 'F5', scale_factor=0.45)

        # Assets
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        nut = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nut.svg")
        
        # Initial placements
        self.place_at_grid(squirrel, 'B2', scale_factor=0.2)
        self.place_at_grid(nut, 'C5', scale_factor=0.2)
        
        epsilon_rect = Rectangle(color="#FFD700", height=0.5, width=4)
        delta_rect = Rectangle(color="#00FFFF", height=4, width=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(squirrel), FadeIn(nut))
        self.play(squirrel.animate.move_to(axes.c2p(3.5, func(3.5))), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.play(squirrel.animate.move_to(axes.c2p(2, func(2))), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.place_at_grid(epsilon_rect, 'B4', scale_factor=0.5)
        self.play(FadeIn(epsilon_rect))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        self.place_at_grid(delta_rect, 'E3', scale_factor=0.5)
        self.play(FadeIn(delta_rect))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.play(Indicate(epsilon_rect), Indicate(delta_rect))
