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
        self.setup_layout("Synthesis and Application", [
            "Binary is an instruction set for recursive puzzles.",
            "Logical pulse controls every disk transition.",
            "This rhythm scales to any number of disks."
        ])
        
        # Load assets
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        tower_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg")
        
        # Display instruction block
        instruction_label = Text("Binary Code", font_size=32, color=WHITE)
        self.place_at_grid(instruction_label, 'A4')
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(Write(instruction_label))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        
        # Rhythmic pulsing wave
        pulse = Annulus(inner_radius=0.2, outer_radius=0.4, color="#00FFFF", fill_opacity=0.5)
        self.place_at_grid(pulse, 'C4')
        self.play(Flash(pulse, color="#00FFFF", flash_radius=1.0))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        
        # Disk multiplying into tower
        disks = VGroup(*[disk_icon.copy() for _ in range(3)])
        disks.arrange(UP, buff=0.1)
        self.place_at_grid(disks, 'D4', scale_factor=0.6)
        
        tower = tower_icon.copy().set_color("#00FF00")
        self.place_at_grid(tower, 'F4', scale_factor=0.8)
        
        self.play(FadeIn(disks))
        self.play(Transform(disks, tower))
