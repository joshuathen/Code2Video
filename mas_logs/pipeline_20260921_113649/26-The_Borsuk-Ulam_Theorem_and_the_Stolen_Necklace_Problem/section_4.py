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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We map discrete cuts to a sphere.",
            "Fair division becomes a zero-sum function.",
            "Borsuk-Ulam guarantees an exact fair solution."
        ]
        self.setup_layout("The Mathematical Bridge", lecture_lines)
        
        # Elements
        necklace = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg", color=YELLOW)
        sphere = Circle(radius=1.2, color=BLUE)
        point = Dot(color=RED)

        # === Animation for Lecture Line 1 ===
        # We map discrete cuts to a sphere.
        self.place_in_area(necklace, "B4", "B6", scale_factor=0.6)
        self.play(FadeIn(necklace))
        self.wait(1)
        
        necklace_loop = Arc(radius=0.6, start_angle=0, angle=2*PI, color=YELLOW)
        self.play(Transform(necklace, necklace_loop))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fair division becomes a zero-sum function.
        self.place_in_area(sphere, "D4", "F6", scale_factor=0.8)
        self.play(Create(sphere))
        
        self.place_at_grid(point, "C3", scale_factor=0.5)
        self.play(FadeIn(point))
        
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Borsuk-Ulam guarantees an exact fair solution.
        # Reuse necklace asset as reference
        ref_necklace = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg", color=GREEN).scale(0.3)
        ref_necklace.move_to(self.grid["C3"])
        self.play(FadeIn(ref_necklace))
        
        self.play(point.animate.move_to(self.grid["D5"]))
        self.play(point.animate.set_color(GREEN))
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
