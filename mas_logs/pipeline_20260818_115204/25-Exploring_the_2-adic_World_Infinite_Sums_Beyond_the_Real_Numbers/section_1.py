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
            "Real numbers measure distance by absolute difference.",
            "2-adic metrics define closeness via divisibility by 2.",
            "Powers of 2 get closer to zero.",
            "High powers of 2 represent very small distances.",
            "This perspective transforms our geometric intuition."
        ]
        self.setup_layout("The Intuition: Re-imagining Distance", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Load Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=WHITE)
        caliper = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/caliper.svg", color=WHITE)
        
        # Create a number line
        line = NumberLine(x_range=[0, 9, 1], length=5, include_numbers=True)
        self.place_in_area(line, 'C1', 'C4', scale_factor=1.0)
        self.place_at_grid(ruler, 'B6', scale_factor=0.3)
        
        # Points
        pts = VGroup(*[Dot(line.n2p(i), color=WHITE) for i in [0, 1, 2, 4, 8]])
        self.play(Create(line), Create(pts), FadeIn(ruler))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        origin_dot = pts[0]
        self.play(origin_dot.animate.set_color(RED))
        self.play(self.lecture[1].animate.set_color(BLUE))

        # === Animation for Lecture Line 3 ===
        dist_8 = MathTex(r"|8-0|_2 = 1/8", color=GREEN).scale(0.8)
        self.place_at_grid(dist_8, 'D2', scale_factor=0.8)
        self.play(FadeIn(dist_8))
        self.play(self.lecture[2].animate.set_color(GREEN))

        # === Animation for Lecture Line 4 ===
        dist_1 = MathTex(r"|1-0|_2 = 1", color=GREEN).scale(0.8)
        self.place_at_grid(dist_1, 'E2', scale_factor=0.8)
        self.play(FadeIn(dist_1))
        self.play(self.lecture[3].animate.set_color(YELLOW))

        # === Animation for Lecture Line 5 ===
        # Shrinking animation simulation using asset
        self.place_at_grid(caliper, 'B5', scale_factor=0.4)
        self.play(FadeIn(caliper))
        self.play(self.lecture[4].animate.set_color(ORANGE))
        self.wait(1)
