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
        # Initialize Scene
        lecture_lines = [
            "Every disk corresponds to a specific binary bit.",
            "Disk one is the rightmost, least significant bit.",
            "Moving n disks requires two to the n minus one moves.",
            "The current move number identifies which disk to move.",
            "Watch as move numbers map to specific disks."
        ]
        self.setup_layout("The Mapping: Disks as Bits", lecture_lines)
        
        # Color definitions from storyboard
        DISK_COLOR = "#90EE90"  # Light Green
        BIT_COLOR = "#00FFFF"   # Cyan
        FORMULA_COLOR = "#FFFFFF" # White
        ASSET_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"

        # === Animation for Lecture Line 1 ===
        # "Every disk corresponds to a specific binary bit."
        self.lecture[0].set_color(WHITE)
        
        # Visual representations of disks using Assets
        disk1 = SVGMobject(ASSET_PATH).set_color(DISK_COLOR).set_fill(DISK_COLOR, opacity=0.8)
        disk2 = SVGMobject(ASSET_PATH).set_color(DISK_COLOR).set_fill(DISK_COLOR, opacity=0.8)
        disk3 = SVGMobject(ASSET_PATH).set_color(DISK_COLOR).set_fill(DISK_COLOR, opacity=0.8)
        
        # Labels for disks
        l1 = Text("Disk 1", font_size=18, color=DISK_COLOR)
        l2 = Text("Disk 2", font_size=18, color=DISK_COLOR)
        l3 = Text("Disk 3", font_size=18, color=DISK_COLOR)
        
        # Positioning: disks in column 2, labels in column 3
        self.place_at_grid(disk1, "B2", scale_factor=0.4)
        self.place_at_grid(l1, "B3")
        self.place_at_grid(disk2, "C2", scale_factor=0.5)
        self.place_at_grid(l2, "C3")
        self.place_at_grid(disk3, "D2", scale_factor=0.6)
        self.place_at_grid(l3, "D3")
        
        self.play(
            FadeIn(disk1), Write(l1),
            FadeIn(disk2), Write(l2),
            FadeIn(disk3), Write(l3),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Disk one is the rightmost, least significant bit."
        self.lecture[1].set_color(BIT_COLOR)
        
        # Mapping labels: "Disk X -> Bit Y"
        m1 = Text("Disk 1 -> Bit 0", font_size=18, color=BIT_COLOR)
        m2 = Text("Disk 2 -> Bit 1", font_size=18, color=BIT_COLOR)
        m3 = Text("Disk 3 -> Bit 2", font_size=18, color=BIT_COLOR)
        
        # Positioning: mapping labels in column 5
        self.place_at_grid(m1, "B5")
        self.place_at_grid(m2, "C5")
        self.place_at_grid(m3, "D5")
        
        arrow1 = Arrow(start=l1.get_right(), end=m1.get_left(), color=BIT_COLOR, buff=0.1)
        arrow2 = Arrow(start=l2.get_right(), end=m2.get_left(), color=BIT_COLOR, buff=0.1)
        arrow3 = Arrow(start=l3.get_right(), end=m3.get_left(), color=BIT_COLOR, buff=0.1)
        
        self.play(Create(arrow1), Write(m1))
        self.play(Create(arrow2), Write(m2), Create(arrow3), Write(m3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Moving n disks requires two to the n minus one moves."
        self.lecture[2].set_color(FORMULA_COLOR)
        
        formula = MathTex(r"\text{Total Moves} = 2^n - 1", color=FORMULA_COLOR)
        # Fix: Adjusted to cover A2-A6 for better alignment
        self.place_in_area(formula, "A2", "A6", scale_factor=0.9)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The current move number identifies which disk to move."
        self.lecture[3].set_color(DISK_COLOR)
        
        binary_info_text = Text("Move Number in Binary:", font_size=18, color=WHITE)
        binary_info_math = MathTex(r"\dots b_2 b_1 b_0", color=BIT_COLOR)
        binary_info = VGroup(binary_info_text, binary_info_math).arrange(DOWN)
        self.place_in_area(binary_info, "E2", "F5", scale_factor=0.8)
        
        self.play(FadeIn(binary_info))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Watch as move numbers map to specific disks."
        self.lecture[4].set_color(WHITE)
        
        # Highlights to emphasize the mapping
        self.play(
            Indicate(disk1, color=BIT_COLOR),
            Indicate(m1, color=WHITE),
            run_time=0.8
        )
        self.play(
            Indicate(disk2, color=BIT_COLOR),
            Indicate(m2, color=WHITE),
            run_time=0.8
        )
        self.play(
            Indicate(disk3, color=BIT_COLOR),
            Indicate(m3, color=WHITE),
            run_time=0.8
        )
        self.wait(2)
