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
        self.setup_layout("Defining the Mystery", [
            "The Brachistochrone problem seeks the fastest descent curve.",
            "Distance is short, but gravity dictates speed.",
            "Imagine a marble rolling down three different tracks."
        ])
        
        # Load assets
        marble_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
        marble_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
        
        label_a = Text("A", font_size=20)
        label_b = Text("B", font_size=20)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(marble_a, 'B2', scale_factor=0.3)
        self.place_at_grid(marble_b, 'D5', scale_factor=0.3)
        self.place_at_grid(label_a, 'B3', scale_factor=0.7)
        self.place_at_grid(label_b, 'F3', scale_factor=0.7)
        self.play(FadeIn(marble_a), FadeIn(marble_b), FadeIn(label_a), FadeIn(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        path = Line(marble_a.get_center(), marble_b.get_center(), color=WHITE)
        tracks_group = VGroup(path)
        self.place_in_area(tracks_group, 'C3', 'E5', scale_factor=0.9)
        self.play(Create(path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        arc1 = ArcBetweenPoints(marble_a.get_center(), marble_b.get_center(), angle=-TAU/6, color=RED)
        arc2 = ArcBetweenPoints(marble_a.get_center(), marble_b.get_center(), angle=TAU/6, color=ORANGE)
        tracks_group.add(arc1, arc2)
        self.play(Create(arc1), Create(arc2))
        self.wait(2)
