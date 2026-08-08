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
        self.setup_layout("The Holographic Recording Process", [
            "A laser beam splits into two paths.",
            "The object beam captures reflected light.",
            "The reference beam provides the phase baseline.",
            "The interference pattern is recorded on film.",
            "This pattern encodes both phase and amplitude."
        ])
        
        # Load assets
        laser_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        object_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/object.svg")
        film_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/film.svg")
        
        # === Animation for Lecture Line 1 ===
        # Laser: self.place_at_grid(laser, 'B3', scale_factor=0.7)
        self.place_at_grid(laser_icon, 'B3', scale_factor=0.7)
        self.play(FadeIn(laser_icon))
        self.lecture[0].set_color("#00FF00")
        
        # === Animation for Lecture Line 2 ===
        # Splitter: self.place_at_grid(splitter, 'D3', scale_factor=0.7)
        splitter = Dot(color=WHITE)
        self.place_at_grid(splitter, 'D3', scale_factor=0.7)
        self.place_at_grid(object_icon, 'D4', scale_factor=0.7)
        obj_beam = Line(start=splitter.get_center(), end=UP*1+RIGHT*2, color=BLUE)
        self.play(Create(splitter), FadeIn(object_icon), Create(obj_beam))
        self.lecture[1].set_color("#0000FF")

        # === Animation for Lecture Line 3 ===
        ref_beam = Line(start=splitter.get_center(), end=DOWN*1+RIGHT*2, color=RED)
        self.play(Create(ref_beam))
        self.lecture[2].set_color("#FF0000")

        # === Animation for Lecture Line 4 ===
        # Film: self.place_at_grid(film, 'C5', scale_factor=0.7)
        self.place_at_grid(film_icon, 'C5', scale_factor=0.7)
        pattern = VGroup(*[Dot(radius=0.03, color="#FF00FF") for _ in range(20)])
        pattern.arrange_in_grid(rows=4, cols=5)
        pattern.move_to(film_icon.get_center())
        self.play(FadeIn(film_icon), FadeIn(pattern))
        self.lecture[3].set_color("#FF00FF")

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(pattern))
        self.lecture[4].set_color("#FFFF00")
        self.wait(2)
