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
        self.setup_layout("The Borsuk-Ulam Theorem", [
            "Borsuk-Ulam connects topology to reality.",
            "Any continuous function maps antipodes equally.",
            "Consider temperature and pressure on Earth.",
            "Two opposite points always match perfectly.",
            "This theorem guarantees a fair solution."
        ])
        
        # Use SVGMobject for Earth asset
        earth_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/earth.svg"
        earth = SVGMobject(earth_asset, fill_color=WHITE)
        self.place_at_grid(earth, 'C4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(earth))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        p1 = Dot(color="#00FF00")
        p2 = Dot(color="#00FF00")
        # antipodal points on the earth icon
        p1.move_to(earth.get_center() + np.array([0.8, 0, 0]))
        p2.move_to(earth.get_center() - np.array([0.8, 0, 0]))
        self.play(FadeIn(p1), FadeIn(p2))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(ORANGE)
        label = Text("T, P", font_size=20)
        self.place_at_grid(label, 'E4', scale_factor=0.8)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        target = Dot(color=RED)
        self.place_at_grid(target, 'B3', scale_factor=0.8)
        self.play(p1.animate.move_to(target.get_center()), p2.animate.move_to(target.get_center()))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        # Using earth asset as background for theorem as requested
        theorem_bg = SVGMobject(earth_asset, fill_color=WHITE, fill_opacity=0.2)
        self.place_at_grid(theorem_bg, 'E5', scale_factor=0.3)
        theorem = Text("f(x) = f(-x)", font_size=24, color=WHITE)
        self.place_at_grid(theorem, 'E5', scale_factor=1.0)
        self.play(FadeIn(theorem_bg), Write(theorem))
        self.wait(2)
