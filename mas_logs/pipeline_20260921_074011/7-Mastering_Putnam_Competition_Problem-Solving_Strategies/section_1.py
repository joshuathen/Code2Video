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
        self.setup_layout("Introduction: The Putnam Mindset", [
            "The Putnam rewards creativity, not brute force.",
            "Focus on core strategies like reduction.",
            "Apply symmetry to simplify complex problems.",
            "Use parity to reveal hidden structures.",
            "Creative proof construction is your goal."
        ])
        
        # Assets
        hammer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hammer.svg")
        mirror = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")
        scale = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(self.lecture[0]))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.place_at_grid(hammer, 'B3', scale_factor=0.5)
        self.play(FadeIn(self.lecture[1]), FadeIn(hammer))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.place_at_grid(mirror, 'C3', scale_factor=0.5)
        self.play(FadeIn(self.lecture[2]), FadeIn(mirror))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.place_at_grid(scale, 'D3', scale_factor=0.5)
        self.play(FadeIn(self.lecture[3]), Flash(scale))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF9900")
        self.place_at_grid(prism, 'E3', scale_factor=0.5)
        self.play(FadeIn(self.lecture[4]), Indicate(prism))
        self.wait(2)
