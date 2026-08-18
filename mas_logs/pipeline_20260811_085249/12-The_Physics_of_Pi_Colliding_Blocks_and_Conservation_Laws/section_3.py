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
        self.setup_layout("Mapping Collisions to Phase Space", [
            "Track velocities as coordinates on phase space.",
            "The collision path traces a circular arc.",
            "Boundary walls force the path to reflect."
        ])
        
        # Axes
        axes = Axes(x_range=[-2, 6, 1], y_range=[-2, 6, 1], axis_config={"include_tip": True, "color": "#808080"})
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        
        # State point
        point = Dot(axes.c2p(2, 2), color="#FF9900")
        self.place_at_grid(point, 'D4', scale_factor=0.6)
        
        # Trajectory (arc)
        arc = ArcBetweenPoints(axes.c2p(2, 2), axes.c2p(4, 0), angle=-TAU/4, color="#00FF00")
        self.place_in_area(arc, 'C3', 'E5', scale_factor=0.5)

        # Asset: Wall
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        wall.set_color("#FFFF00")
        self.place_at_grid(wall, 'A1', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), FadeIn(point))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(arc))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(wall))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
