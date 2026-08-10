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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "For one point, we have one region.",
            "For two points, we get two regions.",
            "Three points give us four regions.",
            "Four points result in eight regions.",
            "Five points create exactly sixteen regions."
        ]
        self.setup_layout("Data Collection: Observing the Pattern", lecture_lines)
        
        # Colors for lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # Assets
        circle_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        self.place_in_area(circle_svg, 'B4', 'E6', scale_factor=0.6)
        
        # region_texts placeholder
        region_texts = []
        for i, val in enumerate([1, 2, 4, 8, 16]):
            t = Text(f"R = {val}", font_size=32, color=colors[i])
            region_texts.append(t)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.add(circle_svg)
        dot1 = Dot(circle_svg.get_critical_point(RIGHT), color=colors[0])
        self.play(FadeIn(dot1))
        self.place_at_grid(region_texts[0], 'B1', scale_factor=0.7)
        self.play(FadeIn(region_texts[0]))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        dot2 = Dot(circle_svg.get_critical_point(LEFT), color=colors[1])
        chord = Line(dot1.get_center(), dot2.get_center(), color=WHITE)
        self.play(FadeIn(dot2), Create(chord))
        self.place_at_grid(region_texts[1], 'B2', scale_factor=0.7)
        self.play(FadeIn(region_texts[1]))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        dot3 = Dot(circle_svg.get_critical_point(UP), color=colors[2])
        c1 = Line(dot1.get_center(), dot3.get_center(), color=WHITE)
        c2 = Line(dot2.get_center(), dot3.get_center(), color=WHITE)
        self.play(FadeIn(dot3), Create(c1), Create(c2))
        self.place_at_grid(region_texts[2], 'B3', scale_factor=0.7)
        self.play(FadeIn(region_texts[2]))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        dot4 = Dot(circle_svg.get_center() + RIGHT * 0.5 + UP * 0.5, color=colors[3])
        self.play(FadeIn(dot4))
        self.place_at_grid(region_texts[3], 'B4', scale_factor=0.7)
        self.play(FadeIn(region_texts[3]))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        dot5 = Dot(circle_svg.get_center() + LEFT * 0.5 + DOWN * 0.5, color=colors[4])
        self.play(FadeIn(dot5))
        self.place_at_grid(region_texts[4], 'B5', scale_factor=0.7)
        self.play(FadeIn(region_texts[4]))
        
        self.wait(2)
