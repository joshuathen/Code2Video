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
        lecture_lines = ["Move disks between three pegs.", "Never place large disks on small ones.", "The minimum moves follow two to the n minus one.", "A three-disk stack takes seven moves.", "The puzzle reveals a recursive pattern."]
        self.setup_layout("The Towers of Hanoi Challenge", lecture_lines)
        
        # Assets
        peg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        disk_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # Create Pegs and Disks using SVGMobjects from Assets
        peg_a = SVGMobject(peg_path).set_color(BLUE)
        peg_b = SVGMobject(peg_path).set_color(BLUE)
        peg_c = SVGMobject(peg_path).set_color(BLUE)
        pegs = VGroup(peg_a, peg_b, peg_c).arrange(RIGHT, buff=1.0)
        
        # 3 disks
        disk1 = SVGMobject(disk_path).set_color(RED).scale(0.5)
        disk2 = SVGMobject(disk_path).set_color(GREEN).scale(0.7)
        disk3 = SVGMobject(disk_path).set_color(BLUE).scale(0.9)
        stack = VGroup(disk3, disk2, disk1).arrange(UP, buff=0.05)
        stack.next_to(peg_a.get_bottom(), UP, buff=0.1)
        
        full_puzzle = VGroup(pegs, stack)
        
        # Position using layout rules
        self.place_in_area(full_puzzle, 'D3', 'F6', scale_factor=0.45)
        
        min_steps_text = MathTex(r"Moves = 2^n - 1", color=WHITE)
        self.place_at_grid(min_steps_text, 'B5', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(full_puzzle), FadeIn(min_steps_text))
        self.lecture[0].set_color(BLUE)
        
        # === Animation for Lecture Line 2 ===
        self.play(Indicate(stack, color=RED))
        self.lecture[1].set_color(RED)
        
        # === Animation for Lecture Line 3 ===
        self.play(Flash(min_steps_text))
        self.lecture[2].set_color(YELLOW)
        
        # === Animation for Lecture Line 4 ===
        # Simple movement of disk stack
        self.play(stack.animate.move_to(peg_c.get_bottom() + UP*0.15))
        self.lecture[3].set_color(ORANGE)
        
        # === Animation for Lecture Line 5 ===
        # Reveal recursive pattern
        final_stack = VGroup(disk3.copy(), disk2.copy(), disk1.copy()).arrange(UP, buff=0.05)
        final_stack.next_to(peg_c.get_bottom(), UP, buff=0.1)
        self.play(FadeIn(final_stack))
        self.lecture[4].set_color(PURPLE)
        self.wait(2)
