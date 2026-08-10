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
        self.setup_layout("The Towers of Hanoi Puzzle", ["Three rods hold stacked disks.", "Smaller disks must stay above larger ones.", "Move all disks to the final rod."])
        
        # Load assets
        rod_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg"
        disk_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # Create mobjects from assets
        rods = VGroup(*[SVGMobject(rod_asset, color=GREY) for _ in range(3)])
        rods.arrange(RIGHT, buff=1.5)
        
        # Define disks
        disks = VGroup(*[SVGMobject(disk_asset, color=WHITE) for _ in range(3)])
        disk_stack = VGroup(disks[2], disks[1], disks[0]).arrange(UP, buff=0.1)
        
        # Use place_in_area for rods as per feedback
        self.place_in_area(rods, 'B3', 'C5', scale_factor=0.85)
        
        # === Animation for Lecture Line 1 ===
        # Use place_at_grid for disk_stack as per feedback
        self.place_at_grid(disk_stack, 'C3', scale_factor=0.8)
        self.play(self.lecture[0].animate.set_color("#E74C3C"), Create(disk_stack))

        # === Animation for Lecture Line 2 ===
        # Show invalid move (red flash) and then correct stacking (green)
        self.play(self.lecture[1].animate.set_color("#2ECC71"),
                  disks[0].animate.set_color("#E74C3C").shift(RIGHT*1.5), 
                  run_time=1)
        self.play(disks[0].animate.set_color("#2ECC71").move_to(self.grid['C4']))

        # === Animation for Lecture Line 3 ===
        # Animate disk movement (purple)
        self.play(self.lecture[2].animate.set_color("#9B59B6"),
                  disks[1].animate.move_to(self.grid['C5']))
        
        self.wait(1)
