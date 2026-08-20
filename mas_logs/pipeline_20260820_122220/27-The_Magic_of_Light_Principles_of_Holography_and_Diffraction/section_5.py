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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Real-World Application", [
            "Holography is precise control of light diffraction.",
            "We store complete light wave physics.",
            "This transforms imaging and secure authentication."
        ])
        
        self.play(FadeIn(self.lecture))

        # === Animation for Lecture Line 1 ===
        # Holography is precise control of light diffraction.
        line1_color = "#FFFFFF"
        self.lecture[0].set_color(line1_color)
        
        # Visual: Wavefronts diffracting
        wave = VGroup(*[Circle(radius=0.3 + i*0.1, color=line1_color).move_to(self.grid["C3"]) for i in range(3)])
        self.play(Create(wave))

        # === Animation for Lecture Line 2 ===
        # We store complete light wave physics.
        line2_color = "#00FFFF"
        self.lecture[1].set_color(line2_color)
        
        # Visual: 3D cube representation
        box = Cube(side_length=1.5, fill_opacity=0.3, color=line2_color)
        # Fix from issue 32/47: use C5 instead of D5, scale 0.6
        self.place_at_grid(box, "C5", scale_factor=0.6)
        
        # Asset: Show real-world holographic application
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg]
        hologram_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg", color=line2_color)
        
        # Combining into a group and applying layout fix (issue 34/49)
        group_elements = VGroup(box, hologram_icon)
        self.place_in_area(group_elements, "B3", "D4", scale_factor=0.7)
        
        self.play(FadeIn(group_elements))

        # === Animation for Lecture Line 3 ===
        # This transforms imaging and secure authentication.
        line3_color = "#FFD700"
        self.lecture[2].set_color(line3_color)
        
        # Visual: Shield/Secure icon
        shield = Star(n=5, color=line3_color)
        # Fix from issue 33/48: B4 instead of B5, scale 0.7
        self.place_at_grid(shield, "B4", scale_factor=0.7)
        self.play(DrawBorderThenFill(shield))
        
        self.wait(2)
