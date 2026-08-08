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
        lecture_lines_text = [
            "Which path minimizes the time between these points?",
            "Gravity pulls objects down a wire path.",
            "Compare three paths: straight, arc, and curve.",
            "Which bead reaches the end first?",
            "The answer is not a straight line."
        ]
        self.setup_layout("The Hook: The Race of Gravity", lecture_lines_text)
        
        # Assets
        ball_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"
        bead_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg"
        
        ball = SVGMobject(ball_path, fill_color="#FFD700", fill_opacity=1)
        bead = SVGMobject(bead_path, fill_color="#FFD700", fill_opacity=1)
        
        # Define paths
        start = self.grid["B1"]
        end = self.grid["E6"]
        
        straight_path = Line(start, end, color=WHITE)
        arc_path = ArcBetweenPoints(start, end, angle=-TAU/8, color=WHITE)
        curve_path = CubicBezier(
            start, 
            start + np.array([1, 1, 0]), 
            end + np.array([-1, 1, 0]), 
            end, 
            color=WHITE
        )
        
        # Group them as race_path_graphic for fix 22
        race_paths = VGroup(straight_path, arc_path, curve_path)
        
        self.add(race_paths)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Display ball at start
        self.place_at_grid(ball, "B1", scale_factor=0.3)
        self.play(FadeIn(ball))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        # Show gravity concept
        gravity_arrow = Arrow(UP, DOWN, color=WHITE).scale(0.5)
        self.place_at_grid(gravity_arrow, "C3", scale_factor=0.8)
        self.play(Create(gravity_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.place_in_area(race_paths, "D1", "F6", scale_factor=0.6)
        self.play(FadeIn(race_paths))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        # Racing balls
        b1 = ball.copy()
        b2 = ball.copy()
        b3 = ball.copy()
        
        self.play(
            MoveAlongPath(b1, straight_path, run_time=3, rate_func=linear),
            MoveAlongPath(b2, arc_path, run_time=2.5, rate_func=linear),
            MoveAlongPath(b3, curve_path, run_time=1.5, rate_func=linear)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF6347"))
        self.place_at_grid(bead, "E6", scale_factor=0.4)
        self.play(ReplacementTransform(b3, bead))
        self.play(Indicate(curve_path, color="#00FFFF"))
        self.wait(2)
