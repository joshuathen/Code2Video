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
        # Title and Lecture Lines
        self.setup_layout("The Legend and the Logic", [
            "Meet the Towers of Hanoi puzzle: three pegs, 'n' disks.",
            "Move all disks to the final peg following two rules.",
            "One disk at a time; never place larger on smaller."
        ])
        
        # Colors
        COLOR_PEG = "#FFFFFF"
        COLOR_SMALL = "#FF5733"
        COLOR_MED = "#33FF57"
        COLOR_LARGE = "#3357FF"
        COLOR_CROSS = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Meet the Towers of Hanoi puzzle: three pegs, 'n' disks.
        self.lecture[0].set_color(YELLOW)
        
        # Create Pegs using Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg]
        peg_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg").set_color(COLOR_PEG)
        peg_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg").set_color(COLOR_PEG)
        peg_c = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg").set_color(COLOR_PEG)
        
        # Positioning pegs to cover the vertical span B to E in columns 2, 4, 6
        self.place_in_area(peg_a, "B2", "E2", scale_factor=1.4)
        self.place_in_area(peg_b, "B4", "E4", scale_factor=1.4)
        self.place_in_area(peg_c, "B6", "E6", scale_factor=1.4)
        
        # Labels for Pegs
        label_a = Text("A", font_size=24, color=COLOR_PEG)
        label_b = Text("B", font_size=24, color=COLOR_PEG)
        label_c = Text("C", font_size=24, color=COLOR_PEG)
        self.place_at_grid(label_a, "F2")
        self.place_at_grid(label_b, "F4")
        self.place_at_grid(label_c, "F6")

        # Disks (Placed on Peg A initially)
        # RoundedRectangle for disks
        disk_large = RoundedRectangle(corner_radius=0.1, width=1.8, height=0.6, color=COLOR_LARGE, fill_opacity=1)
        disk_med = RoundedRectangle(corner_radius=0.1, width=1.3, height=0.6, color=COLOR_MED, fill_opacity=1)
        disk_small = RoundedRectangle(corner_radius=0.1, width=0.8, height=0.6, color=COLOR_SMALL, fill_opacity=1)

        # Place disks on Peg A (Column 2)
        # Fixes from VideoCritic (Issues 21, 22)
        self.place_in_area(disk_large, 'E1', 'E3')
        self.place_in_area(disk_med, 'D1', 'D3', scale_factor=0.7)
        self.place_at_grid(disk_small, "C2")

        # Execution
        self.play(
            FadeIn(peg_a), FadeIn(peg_b), FadeIn(peg_c),
            Write(label_a), Write(label_b), Write(label_c)
        )
        self.play(FadeIn(disk_large), FadeIn(disk_med), FadeIn(disk_small))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Move all disks to the final peg following two rules.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Animation: Move Smallest disk from Peg A to Peg C
        # It goes from C2 to E6
        self.play(disk_small.animate.move_to(self.grid["E6"]))
        self.wait(0.5)
        
        # Animation: Move Medium disk from Peg A to Peg B
        # It goes from D2 to E4
        self.play(disk_med.animate.move_to(self.grid["E4"]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # One disk at a time; never place larger on smaller.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Invalid Move Attempt: Large disk tries to go on top of Medium disk on Peg B
        # Target position is above medium disk on Peg B (D4)
        target_pos_invalid = self.grid["D4"]
        
        # Create a red cross
        cross = Cross(stroke_color=COLOR_CROSS, stroke_width=20)
        # Fix from VideoCritic (Issue 20)
        self.place_at_grid(cross, 'D4', scale_factor=0.9)

        # Large disk moves to Peg B
        self.play(disk_large.animate.move_to(target_pos_invalid))
        self.play(Create(cross))
        self.wait(1)
        
        # Show failure and return
        self.play(FadeOut(cross), disk_large.animate.move_to(self.grid["E2"]))
        
        self.lecture[2].set_color(WHITE)
        self.wait(2)
