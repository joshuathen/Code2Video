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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mapping Moves to Binary", ["Binary numbers dictate disk moves.", "Rightmost one shows disk size.", "Binary sequence solves the puzzle."])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        rod_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg", color=WHITE)
        rod_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg", color=WHITE)
        rod_c = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg", color=WHITE)
        
        self.place_at_grid(rod_a, 'C2', scale_factor=0.6)
        self.place_at_grid(rod_b, 'C4', scale_factor=0.6)
        self.place_at_grid(rod_c, 'C6', scale_factor=0.6)
        
        label_a = Text("A", font_size=20, color="#FF4500").next_to(rod_a, DOWN)
        label_b = Text("B", font_size=20, color="#FF4500").next_to(rod_b, DOWN)
        label_c = Text("C", font_size=20, color="#FF4500").next_to(rod_c, DOWN)
        
        self.play(FadeIn(rod_a, rod_b, rod_c), FadeIn(label_a, label_b, label_c))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#ADFF2F")
        disk_zero = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg", color="#ADFF2F")
        bit_zero = Text("0", color="#ADFF2F").scale(0.7)
        bit_zero.next_to(disk_zero, UP)
        bit_group_zero = VGroup(disk_zero, bit_zero)
        
        self.place_at_grid(bit_group_zero, 'E3', scale_factor=0.8)
        self.play(Write(bit_group_zero))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF69B4")
        disk_one = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg", color="#FF69B4")
        bit_one = Text("1", color="#FF69B4").scale(0.7)
        bit_one.next_to(disk_one, UP)
        bit_group_one = VGroup(disk_one, bit_one)
        
        self.place_at_grid(bit_group_one, 'E5', scale_factor=0.8)
        self.play(Write(bit_group_one))
        
        self.wait(2)
