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
        self.setup_layout("The Bat's Echolocation", [
            "Bats balance time and frequency resolution.",
            "Short clicks aid distance detection.",
            "Frequency sweeps help identify prey."
        ])
        
        # Assets
        bat = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bat.svg")
        target = Circle(radius=0.2, color=YELLOW, fill_opacity=0.5)
        combined_group = VGroup(bat, target)
        
        # Layout
        self.place_at_grid(bat, 'C3', scale_factor=0.6)
        self.place_at_grid(target, 'D5', scale_factor=0.7)
        self.place_in_area(combined_group, 'C3', 'E5', scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        # Emit frequency-modulated chirp
        chirp = VGroup(*[Line(bat.get_right(), bat.get_right()+RIGHT*0.5, color=RED) for _ in range(3)])
        chirp.arrange(RIGHT, buff=0.1)
        self.play(FadeIn(chirp), chirp.animate.shift(RIGHT*2))
        self.play(FadeOut(chirp))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        # Show echo returning
        pulse = Dot(color=RED).move_to(bat.get_right())
        self.play(pulse.animate.move_to(target.get_center()))
        self.play(pulse.animate.move_to(bat.get_center()))
        self.play(FadeOut(pulse))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        # Illustrate time-of-flight / sweep
        sweep = VMobject(color=YELLOW)
        sweep.set_points_smoothly([bat.get_right(), bat.get_right()+RIGHT*1.5+UP*0.5, bat.get_right()+RIGHT*2+DOWN*0.5])
        self.play(Create(sweep))
        self.play(FadeOut(sweep))
