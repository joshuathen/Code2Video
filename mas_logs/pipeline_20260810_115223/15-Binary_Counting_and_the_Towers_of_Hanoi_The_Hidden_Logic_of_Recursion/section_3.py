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
        self.setup_layout("Mapping Moves to Binary", [
            "Each binary count step mirrors one move.",
            "The changing bit dictates the disk moved.",
            "The counter maps perfectly to the puzzle.",
            "We watch the binary counter tick up.",
            "Physical movement aligns with abstract counting."
        ])
        
        # Assets
        disk_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        tower_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg"
        
        # Setup Counter
        counter = Text("000", font_size=48, color=YELLOW)
        self.place_at_grid(counter, "D5", scale_factor=1.0)
        
        # Setup Towers and Disks
        towers = VGroup(*[SVGMobject(tower_path) for _ in range(3)]).arrange(RIGHT, buff=1.0)
        disks = VGroup(*[SVGMobject(disk_path) for _ in range(3)]).arrange(DOWN, buff=0.1)
        
        # Visual Grouping
        puzzle_group = VGroup(towers, disks)
        self.place_in_area(puzzle_group, "B1", "E3", scale_factor=0.7)
        self.place_in_area(disks, "B4", "E5", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        counter_1 = Text("001", font_size=48, color=YELLOW).move_to(counter.get_center())
        # Move smallest disk (top one)
        self.play(ReplacementTransform(counter, counter_1), disks[2].animate.shift(RIGHT))
        counter = counter_1
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        counter_2 = Text("010", font_size=48, color=YELLOW).move_to(counter.get_center())
        # Move second disk
        self.play(ReplacementTransform(counter, counter_2), disks[1].animate.shift(RIGHT))
        counter = counter_2
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        counter_3 = Text("011", font_size=48, color=YELLOW).move_to(counter.get_center())
        # Move smallest disk
        self.play(ReplacementTransform(counter, counter_3), disks[2].animate.shift(RIGHT))
        counter = counter_3
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        # Move largest disk
        self.play(disks[0].animate.shift(RIGHT))
        self.wait(2)
