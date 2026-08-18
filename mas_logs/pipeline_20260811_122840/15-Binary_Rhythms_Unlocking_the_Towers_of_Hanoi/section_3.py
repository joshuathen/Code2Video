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
            "Disk moves match binary counting patterns.",
            "Each bit position controls one specific disk.",
            "Counting in binary reveals the move order."
        ])
        self.lecture.set_opacity(1)

        # Load assets
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")

        # === Animation for Lecture Line 1 ===
        # Represent Binary pattern
        bin_group = VGroup(*[Text(f"{i:03b}", color=YELLOW) for i in range(1, 4)]).arrange(DOWN)
        # Apply layout fix from issue 27/42
        self.place_in_area(bin_group, "A1", "C2", scale_factor=0.7)
        self.play(FadeIn(bin_group))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Mapping bit to disk [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg]
        bit_label = Text("Bit", color=BLUE)
        disk_label = Text("Disk", color=GREEN)
        mapping_line = Arrow(start=UP, end=DOWN, color=WHITE).scale(0.5)
        
        # Integrating Asset disk.svg
        disk_img = disk_icon.copy()
        
        bit_to_disk = VGroup(bit_label, mapping_line, disk_label, disk_img).arrange(DOWN)
        # Apply layout fix from issue 29/44
        self.place_at_grid(bit_to_disk, "A5", scale_factor=0.75)
        self.play(FadeIn(bit_to_disk))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        # Binary Table [Asset: binary_table]
        table = Table(
            [["001", "Disk 1"], ["010", "Disk 2"], ["011", "Disk 1"]],
            col_labels=[Text("Binary"), Text("Move")],
            include_outer_lines=True
        ).scale(0.3)
        # Apply layout fix from issue 28/43
        self.place_in_area(table, "D1", "F3", scale_factor=0.6)
        self.play(Create(table))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
