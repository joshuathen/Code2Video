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
        self.setup_layout("The Phase Space Mapping", [
            "Represent system as point in space.",
            "Collisions become bounces off walls.",
            "Angle relates to mass ratio."
        ])
        
        # Assets
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        ball_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[0, 4], y_range=[0, 4], axis_config={"include_tip": False}).scale(0.6)
        point = Dot(color=YELLOW)
        point.move_to(axes.c2p(1, 1))
        phase_space = VGroup(axes, point, billiard_icon)
        self.place_in_area(phase_space, 'A3', 'D5', scale_factor=0.65)
        self.play(FadeIn(phase_space))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        wall = Line(axes.c2p(0, 0), axes.c2p(0, 4), color=BLUE)
        self.add(wall)
        
        traj = TracedPath(point.get_center, stroke_color=RED, stroke_width=2)
        self.add(traj)
        
        self.play(
            point.animate.move_to(axes.c2p(0, 2)),
            run_time=2
        )
        self.lecture[1].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        arc = Arc(radius=0.5, start_angle=0, angle=PI/4, color=GREEN)
        self.place_at_grid(arc, 'F5', scale_factor=0.8)
        
        self.play(Create(arc), FadeIn(ball_icon.move_to(arc.get_center())))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
