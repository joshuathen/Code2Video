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
        lecture_lines = ["Does the pattern continue?", "Try n equals six.", "We count thirty-one regions.", "Pattern breaks at six points.", "Geometry reveals the true count."]
        self.setup_layout("The Counter-Intuitive Reveal", lecture_lines)
        
        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg]
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(circle, 'B4', 'E6', scale_factor=0.45)
        self.add(circle)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        # Note: Use a Circle mobject to calculate points correctly, then draw overlay lines
        base_circle = Circle(radius=circle.width/2).move_to(circle.get_center())
        points = [base_circle.point_from_proportion(i/6) for i in range(6)]
        chord_group = VGroup()
        for i in range(6):
            for j in range(i + 1, 6):
                chord = Line(points[i], points[j], color=WHITE, stroke_width=2)
                chord_group.add(chord)
        self.play(Create(chord_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        regions_text = Text("31 Regions", font_size=36, color=GREEN)
        self.place_at_grid(regions_text, 'B3', scale_factor=0.9)
        self.play(Write(regions_text))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg] used as overlay
        final_text = Text("Pattern Broken: 31 != 32", font_size=24, color=RED)
        self.place_at_grid(final_text, 'F3', scale_factor=0.8)
        self.play(FadeIn(final_text))
        
        # Adding dot for proximity
        dot = Dot(color=PURPLE)
        self.place_at_grid(dot, 'F6', scale_factor=0.5)
        self.play(FadeIn(dot))
        
        self.wait(2)
