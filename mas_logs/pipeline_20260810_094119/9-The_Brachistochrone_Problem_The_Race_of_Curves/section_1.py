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
        lecture_lines = [
            "What is the fastest slide between two points?",
            "Consider a straight line, a circle, and a curve.",
            "Which shape wins the race under gravity?",
            "[Asset: straight_line_vs_arc_vs_curve_race]",
            "Gravity surprises us with the winning path."
        ]
        self.setup_layout("The Brachistochrone Problem", lecture_lines)
        
        # Elements
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        # Using simple primitives for placeholders based on storyboards
        line = Line(LEFT*1, RIGHT*1, color=WHITE)
        arc = Arc(radius=1, start_angle=PI, angle=PI/2, color=YELLOW)
        curve = FunctionGraph(lambda x: -0.5 * (np.sin(2 * x) + 1), x_range=[-1, 1], color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_at_grid(ball, 'D2', scale_factor=0.6)
        self.add(ball)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        container = VGroup(line, arc, curve).arrange(RIGHT)
        self.place_in_area(container, 'B2', 'B5', scale_factor=0.7)
        self.play(Create(line), Create(arc), Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.play(ball.animate.move_to(curve.get_start()))
        self.play(MoveAlongPath(ball, curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF0000")
        target_marker = Star(color=RED, fill_opacity=1)
        self.place_at_grid(target_marker, 'D5', scale_factor=0.7)
        self.play(FadeIn(target_marker))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.play(Indicate(curve))
        self.wait(2)
