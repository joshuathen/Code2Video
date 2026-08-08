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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout("Directional Parity: Where does the disk go?", [
            "Where should the disk go? Follow the parity rule.",
            "For odd disks, move disk 1 in clockwise steps.",
            "For even disks, move disk 1 in counter-clockwise steps."
        ])
        
        # Assets
        PEG_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        DISK_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # Colors
        PATH_COLOR = "#00FFFF" # Cyan
        HIGHLIGHT_COLOR = "#FFFF00" # Yellow
        PEG_COLOR = "#FFFFFF"
        
        # === Animation for Lecture Line 1 ===
        # Use white color for the first line initially
        self.lecture[0].set_color(WHITE)
        
        # Create three pegs in a circular layout (Updated positions as per Issue 31)
        # Peg A: Top (C4)
        # Peg B: Bottom Right (E6)
        # Peg C: Bottom Left (E2)
        peg_a = SVGMobject(PEG_ASSET).set_color(PEG_COLOR)
        peg_b = SVGMobject(PEG_ASSET).set_color(PEG_COLOR)
        peg_c = SVGMobject(PEG_ASSET).set_color(PEG_COLOR)
        
        self.place_at_grid(peg_a, "C4", scale_factor=0.5)
        self.place_at_grid(peg_b, "E6", scale_factor=0.5)
        self.place_at_grid(peg_c, "E2", scale_factor=0.5)
        
        # Labels for pegs
        label_a = Text("Peg A", font_size=20).next_to(peg_a, UP, buff=0.1)
        label_b = Text("Peg B", font_size=20).next_to(peg_b, RIGHT, buff=0.1)
        label_c = Text("Peg C", font_size=20).next_to(peg_c, LEFT, buff=0.1)
        
        pegs_group = VGroup(peg_a, peg_b, peg_c, label_a, label_b, label_c)
        self.play(FadeIn(pegs_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture colors
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(PATH_COLOR)
        )
        
        # Total Disk Parity Label (Rule Trigger) - Fixed position (Issue 29)
        odd_label = Text("Total Disks: ODD", font_size=24, color=HIGHLIGHT_COLOR)
        self.place_in_area(odd_label, "A1", "A6", scale_factor=0.8)
        self.play(Write(odd_label))
        
        # Clockwise Arrows (A -> B -> C -> A)
        # Curved arrows connecting the peg centers or near-centers
        arrow_ab = CurvedArrow(peg_a.get_right() + RIGHT*0.2, peg_b.get_top() + UP*0.2, radius=-2.5, color=PATH_COLOR)
        arrow_bc = CurvedArrow(peg_b.get_left() + LEFT*0.2, peg_c.get_right() + RIGHT*0.2, radius=-3.5, color=PATH_COLOR)
        arrow_ca = CurvedArrow(peg_c.get_top() + UP*0.2, peg_a.get_left() + LEFT*0.2, radius=-2.5, color=PATH_COLOR)
        
        arrows_cw = VGroup(arrow_ab, arrow_bc, arrow_ca)
        self.play(Create(arrows_cw))
        
        # Small Disk 1 (Asset)
        disk1 = SVGMobject(DISK_ASSET).set_color(PATH_COLOR)
        disk1.scale(0.2)
        disk1.move_to(peg_a.get_center())
        
        self.play(FadeIn(disk1))
        
        # Move disk 1 in Clockwise steps
        self.play(MoveAlongPath(disk1, arrow_ab))
        self.wait(0.2)
        self.play(MoveAlongPath(disk1, arrow_bc))
        self.wait(0.2)
        self.play(MoveAlongPath(disk1, arrow_ca))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture colors
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Update Parity Label to EVEN - Fixed position (Issue 30)
        even_label = Text("Total Disks: EVEN", font_size=24, color=HIGHLIGHT_COLOR)
        self.place_in_area(even_label, "A1", "A6", scale_factor=0.8)
        self.play(ReplacementTransform(odd_label, even_label))
        
        # Reverse Arrows for Counter-Clockwise (A -> C -> B -> A)
        arrow_ac = CurvedArrow(peg_a.get_left() + LEFT*0.2, peg_c.get_top() + UP*0.2, radius=2.5, color=HIGHLIGHT_COLOR)
        arrow_cb = CurvedArrow(peg_c.get_right() + RIGHT*0.2, peg_b.get_left() + LEFT*0.2, radius=3.5, color=HIGHLIGHT_COLOR)
        arrow_ba = CurvedArrow(peg_b.get_top() + UP*0.2, peg_a.get_right() + RIGHT*0.2, radius=2.5, color=HIGHLIGHT_COLOR)
        
        arrows_ccw = VGroup(arrow_ac, arrow_cb, arrow_ba)
        
        self.play(
            FadeOut(arrows_cw),
            Create(arrows_ccw)
        )
        
        # Move disk 1 in Counter-Clockwise steps
        self.play(MoveAlongPath(disk1, arrow_ac))
        self.wait(0.2)
        self.play(MoveAlongPath(disk1, arrow_cb))
        self.wait(0.2)
        self.play(MoveAlongPath(disk1, arrow_ba))
        self.wait(2)
