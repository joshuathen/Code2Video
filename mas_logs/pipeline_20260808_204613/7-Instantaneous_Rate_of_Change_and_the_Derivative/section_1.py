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
        lecture_lines = ["Speed is distance over time.", "A cheetah runs 100 meters in 5 seconds.", "Average speed is 20 meters per second.", "But how fast at one exact moment?", "That is the challenge of instantaneous speed."]
        self.setup_layout("From Average to Instantaneous", lecture_lines)
        
        cheetah_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(cheetah_img, 'A1', scale_factor=0.3)
        self.play(FadeIn(self.title), FadeIn(cheetah_img))

        axes = Axes(x_range=[0, 6, 1], y_range=[0, 120, 20], axis_config={"include_tip": True}).scale(0.5)
        curve = axes.plot(lambda x: 4 * x**2, color=BLUE)
        graph = VGroup(axes, curve)
        self.place_in_area(graph, 'A3', 'F6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]), Create(graph), run_time=1)
        self.lecture[1].set_color(YELLOW)
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        p1 = axes.c2p(1, 4)
        p2 = axes.c2p(5, 100)
        secant = Line(p1, p2, color=RED)
        self.play(FadeIn(self.lecture[2]), Create(secant), run_time=1)
        self.lecture[2].set_color(YELLOW)
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 4 ===
        p3 = axes.c2p(2.5, 25)
        dot = Dot(p3, color=GREEN)
        self.play(FadeIn(self.lecture[3]), Create(dot), run_time=1)
        self.lecture[3].set_color(YELLOW)
        self.lecture[2].set_color(WHITE)

        # === Animation for Lecture Line 5 ===
        cheetah_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(cheetah_2, 'F1', scale_factor=0.3)
        tangent = TangentLine(curve, alpha=0.5, length=3, color=TEAL)
        self.play(FadeIn(self.lecture[4]), FadeOut(secant), FadeIn(cheetah_2), Create(tangent), run_time=1)
        self.lecture[4].set_color(YELLOW)
        self.lecture[3].set_color(WHITE)
        self.wait(2)
