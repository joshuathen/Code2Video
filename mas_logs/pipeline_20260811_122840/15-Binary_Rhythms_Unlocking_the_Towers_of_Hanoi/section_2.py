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
        self.setup_layout("The Towers of Hanoi Challenge", [
            "Three rods, n-disks; a classic challenge.",
            "Larger disks cannot sit on smaller ones.",
            "Exponential growth: more disks mean many steps."
        ])
        
        # Define rod and disk colors
        rod_color = GRAY
        
        # Create base rods (SVGs or Lines)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disks.svg]
        disks_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disks.svg")
        
        # Re-layout per instructions
        rod1 = Line(UP*0.8, DOWN*0.8, color=rod_color, stroke_width=8)
        rod2 = Line(UP*0.8, DOWN*0.8, color=rod_color, stroke_width=8)
        rod3 = Line(UP*0.8, DOWN*0.8, color=rod_color, stroke_width=8)
        
        self.place_in_area(rod1, 'B2', 'E2', scale_factor=0.8)
        self.place_in_area(rod2, 'B3', 'E3', scale_factor=0.8)
        self.place_in_area(rod3, 'B4', 'E4', scale_factor=0.8)
        rods = VGroup(rod1, rod2, rod3)

        # Create disks
        disk1 = Rectangle(width=0.8, height=0.25, fill_opacity=1, color="#FF5733")
        disk2 = Rectangle(width=1.2, height=0.25, fill_opacity=1, color="#33FF57")
        disk3 = Rectangle(width=1.6, height=0.25, fill_opacity=1, color="#3357FF")
        stack = VGroup(disk3, disk2, disk1).arrange(UP, buff=0)
        stack.move_to(rod1.get_bottom() + UP*0.375)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(rods), FadeIn(stack))
        self.add(stack)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(Indicate(stack))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Placeholder for asset hanoi_moves
        hanoi_moves = Text("Steps: 1 -> 3 -> 7", font_size=24)
        self.place_at_grid(hanoi_moves, 'F3', scale_factor=0.6)
        
        # Animate movement
        target_pos = rod3.get_bottom() + UP*0.375
        self.play(
            stack.animate.move_to(target_pos),
            FadeIn(hanoi_moves),
            run_time=2
        )
        self.wait(1)
