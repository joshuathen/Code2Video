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
        self.setup_layout("Mapping Collisions to Geometry", [
            "We map velocities to a 2D plane.",
            "Each collision is a reflection in space.",
            "The point traces an arc of a circle.",
            "Energy conservation keeps the path fixed.",
            "Collisions hit the circle's boundaries perfectly."
        ])
        
        # Setup Axes, Ball, and Billiard icon
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 5, 1], axis_config={"include_tip": True})
        
        ball_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        
        # Apply layout fixes from VideoCritic
        self.place_in_area(axes, 'C3', 'F5', scale_factor=0.6)
        
        dot = self.place_at_grid(Dot(color=YELLOW), 'D3', scale_factor=0.7)
        arc = self.place_in_area(Arc(radius=2, start_angle=0, angle=PI/2, color=BLUE), 'C3', 'E5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE), Create(axes), FadeIn(ball_icon.next_to(axes, UP)), run_time=1.5)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN), Create(arc), run_time=1.5)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW), 
                  FadeIn(billiard_icon.next_to(dot, RIGHT)),
                  UpdateFromAlphaFunc(dot, lambda m, a: m.move_to(axes.c2p(2*np.cos(a*PI/2), 2*np.sin(a*PI/2)))),
                  run_time=2)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED), Indicate(arc), run_time=1)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(ORANGE), FadeOut(arc), run_time=1.5)
        self.wait(1)
