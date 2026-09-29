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
        self.setup_layout("The Microscopic Origin: Induced Dipoles", [
            "Light waves oscillate electrons in atoms.",
            "Oscillating electrons re-radiate waves.",
            "Interference creates a slowed superposition."
        ])
        
        # Elements using assets
        atom_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg")
        electron_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/electron.svg")
        
        atom_group = VGroup(atom_img, electron_img)
        atom_img.set_color("#00FFFF")
        electron_img.set_color("#FF8000")
        
        # Apply layout fix as per VideoCritic (Issue 26)
        self.place_in_area(atom_group, 'C4', 'E6', scale_factor=0.8)
        
        # Animation
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        # Animate electron cloud shift (Issue 17)
        self.play(electron_img.animate.shift(RIGHT * 0.5), run_time=1.5)
        self.play(electron_img.animate.shift(LEFT * 1.0), run_time=1.5)
        self.play(electron_img.animate.shift(RIGHT * 0.5), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8000"))
        # Add a radiating wave representation
        wave = Circle(radius=0.1, color=YELLOW, stroke_width=3)
        self.add(wave)
        wave.add_updater(lambda m, dt: m.set_width(m.width + dt * 2))
        self.play(FadeIn(wave))
        self.wait(2)
        wave.remove_updater(wave.updaters[0])

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeOut(wave))
        self.wait(2)
