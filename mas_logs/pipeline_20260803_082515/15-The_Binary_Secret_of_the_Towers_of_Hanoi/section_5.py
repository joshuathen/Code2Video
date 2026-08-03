from manim import *
import numpy as np

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
        self.setup_layout("The 3-Disk Synchronized Walkthrough", [
            "Move one: Bit one flips, move the smallest disk.",
            "Move two: Bit two flips, move the medium disk.",
            "Move three: Bit one flips, move the smallest disk.",
            "Move four: Bit three flips, move the largest disk.",
            "Follow the binary pulse until the tower is complete."
        ])

        # Colors
        CYAN = "#00FFFF"
        MAGENTA = "#FF00FF"
        GOLD = "#FFD700"
        WHITE_C = "#FFFFFF"
        # Using colors consistent with Section 1: Red (Small), Green (Medium), Blue (Large)
        DISK_COLORS = [RED, GREEN, BLUE] 

        # Disks and Pegs setup
        # Resolution of Issue 37: Scaling pegs and placing at D3, D4, D5
        peg_a = Rectangle(height=3, width=0.1, color=WHITE, fill_opacity=1)
        peg_b = Rectangle(height=3, width=0.1, color=WHITE, fill_opacity=1)
        peg_c = Rectangle(height=3, width=0.1, color=WHITE, fill_opacity=1)
        
        self.place_at_grid(peg_a, "D3", scale_factor=0.7)
        self.place_at_grid(peg_b, "D4", scale_factor=0.7)
        self.place_at_grid(peg_c, "D5", scale_factor=0.7)
        
        pegs = [peg_a, peg_b, peg_c]
        self.add(*pegs)
        
        # Labels for Pegs
        label_a = Text("A", font_size=20).next_to(peg_a, DOWN, buff=0.1)
        label_b = Text("B", font_size=20).next_to(peg_b, DOWN, buff=0.1)
        label_c = Text("C", font_size=20).next_to(peg_c, DOWN, buff=0.1)
        self.add(label_a, label_b, label_c)

        # Disks: 0=Smallest, 1=Medium, 2=Largest
        # Disks scaled by 0.7 to match pegs
        disk3 = RoundedRectangle(corner_radius=0.07, height=0.28, width=1.26, fill_color=DISK_COLORS[2], fill_opacity=1)
        disk2 = RoundedRectangle(corner_radius=0.07, height=0.28, width=0.98, fill_color=DISK_COLORS[1], fill_opacity=1)
        disk1 = RoundedRectangle(corner_radius=0.07, height=0.28, width=0.7, fill_color=DISK_COLORS[0], fill_opacity=1)
        
        disks = [disk1, disk2, disk3]
        
        # Initial positions on Peg A
        peg_stacks = [[disk3, disk2, disk1], [], []]
        
        def get_peg_bottom(peg_index):
            return pegs[peg_index].get_bottom()

        # Position disks on Peg A
        for i, disk in enumerate(peg_stacks[0]):
            disk.move_to(get_peg_bottom(0) + UP * (0.15 + i * 0.3))
        
        self.add(disk1, disk2, disk3)

        # Binary Counter
        # Resolution of Issue 38: Binary bits indicator at B4 with scale_factor=0.7
        bits = VGroup(Text("0", font_size=36), Text("0", font_size=36), Text("0", font_size=36)).arrange(RIGHT, buff=0.2)
        self.place_at_grid(bits, "B4", scale_factor=0.7)
        self.add(bits)

        def update_bits(val_str, current_bits, color=WHITE_C):
            new_bits = VGroup(Text(val_str[0], font_size=36), Text(val_str[1], font_size=36), Text(val_str[2], font_size=36)).arrange(RIGHT, buff=0.2)
            new_bits.scale(0.7) # maintain scale factor
            new_bits.move_to(current_bits.get_center())
            new_bits.set_color(color)
            return new_bits

        def move_disk_anim(disk_idx, from_peg, to_peg, duration=0.8):
            disk = disks[disk_idx]
            peg_stacks[from_peg].remove(disk)
            target_h = len(peg_stacks[to_peg])
            target_pos = get_peg_bottom(to_peg) + UP * (0.15 + target_h * 0.3)
            peg_stacks[to_peg].append(disk)
            return disk.animate(run_time=duration, path_arc=PI/2).move_to(target_pos)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(CYAN))
        new_bits = update_bits("001", bits, color=CYAN)
        self.play(FadeOut(bits), FadeIn(new_bits), move_disk_anim(0, 0, 1, duration=0.6))
        bits = new_bits

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(CYAN))
        new_bits = update_bits("010", bits, color=CYAN)
        self.play(FadeOut(bits), FadeIn(new_bits), move_disk_anim(1, 0, 2, duration=0.6))
        bits = new_bits

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(CYAN))
        new_bits = update_bits("011", bits, color=CYAN)
        self.play(FadeOut(bits), FadeIn(new_bits), move_disk_anim(0, 1, 2, duration=0.6))
        bits = new_bits

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(MAGENTA))
        new_bits = update_bits("100", bits, color=MAGENTA)
        # Apply 1.5x scale from storyboard
        final_bits_4 = new_bits.copy().scale(1.5)
        self.play(
            FadeOut(bits),
            FadeIn(new_bits),
            new_bits.animate.scale(1.5),
            move_disk_anim(2, 0, 1, duration=1.0)
        )
        bits = new_bits # Now bits is scaled 1.5x larger than 0.7 original

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(WHITE_C))
        
        # Move 5 (101)
        new_bits_5 = update_bits("101", bits, color=WHITE_C)
        # Note: update_bits resets scale to 0.7, so we apply the 1.5x scale to keep it consistent if intended, 
        # but the storyboard only mentioned scaling up 1.5x for '100'. 
        # I'll scale it back to 0.7 for the remaining moves as the "pulse" continues.
        self.play(FadeOut(bits), FadeIn(new_bits_5), move_disk_anim(0, 2, 0, duration=0.6))
        bits = new_bits_5
        
        # Move 6 (110)
        new_bits_6 = update_bits("110", bits, color=WHITE_C)
        self.play(FadeOut(bits), FadeIn(new_bits_6), move_disk_anim(1, 2, 1, duration=0.6))
        bits = new_bits_6
        
        # Move 7 (111)
        new_bits_7 = update_bits("111", bits, color=WHITE_C)
        self.play(FadeOut(bits), FadeIn(new_bits_7), move_disk_anim(0, 0, 1, duration=0.6))
        bits = new_bits_7
        
        self.wait(0.5)
        
        # Equation: 2^3 - 1 = 7 pulse
        # Resolution of Issue 39: Final equation at B4 with scale_factor=0.8
        equation = MathTex("2^3 - 1 = 7", color=WHITE_C, font_size=48)
        self.place_at_grid(equation, "B4", scale_factor=0.8)
        
        self.play(FadeOut(bits), Write(equation))
        for _ in range(2):
            self.play(equation.animate.scale(1.2), run_time=0.3)
            self.play(equation.animate.scale(1/1.2), run_time=0.3)
        
        # Completion Flash: Gold outline on tower on Peg B (Peg index 1)
        tower = VGroup(disk3, disk2, disk1)
        glow = tower.copy().set_style(stroke_color=GOLD, stroke_width=8, fill_opacity=0)
        self.play(FadeIn(glow))
        self.play(Flash(tower, color=GOLD, flash_radius=1.2))
        self.play(FadeOut(glow))
        self.wait(2)
        self.play(self.lecture[4].animate.set_color(WHITE))
