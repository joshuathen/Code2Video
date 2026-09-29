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
            "Disks move only between adjacent pegs.",
            "State transitions follow ternary arithmetic increments.",
            "Each move adds one to the state.",
            "The sequence maps linearly on a path.",
            "This constrains the Tower of Hanoi."
        ]
        self.setup_layout("The Constrained Towers of Hanoi", lecture_lines)
        
        # Create elements
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disks.svg]
        disks_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disks.svg")
        disks_icon.set_color("#F1FAEE")
        
        # Adjust layout based on feedback issues 24, 25, 26, 34
        self.place_in_area(disks_icon, 'A2', 'B5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(disks_icon))
        self.lecture[0].set_color("#F1FAEE")
        
        # === Animation for Lecture Line 2 ===
        state_text = MathTex("000_3 \\rightarrow 001_3").scale(0.8)
        self.place_at_grid(state_text, 'D2', scale_factor=0.75)
        self.play(Write(state_text))
        self.lecture[1].set_color("#A8DADC")
        
        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disks.svg]
        self.play(disks_icon.animate.set_color("#A8DADC"))
        self.lecture[2].set_color("#F1FAEE")
        
        # === Animation for Lecture Line 4 ===
        path = NumberLine(x_range=[0, 3, 1], length=4)
        path.add_ticks()
        self.place_at_grid(path, 'E2', scale_factor=0.8)
        self.play(Create(path))
        self.lecture[3].set_color("#A8DADC")
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#F1FAEE")
        self.wait(1)
