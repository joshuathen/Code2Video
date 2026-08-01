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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mapping the Positions (The Binary Address)", [
            "Parity bits live at positions 1, 2, and 4.",
            "Data bits fill the remaining available slots.",
            "Each position is a sum of unique powers.",
            "Position seven is monitored by bits 1, 2, and 4.",
            "Binary addresses tell us exactly where bits live."
        ])
        
        # Colors
        TOWER_COLOR = "#FFFF00"
        CARGO_COLOR = "#00FFFF"
        
        # Assets
        tower_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg", color=TOWER_COLOR)
        cargo_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cargo.svg", color=CARGO_COLOR)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(TOWER_COLOR)
        
        # Create 7 slots with labels
        slots_vgroup = VGroup()
        for i in range(7):
            s = Square(side_length=0.6, color=WHITE)
            l = Text(str(i+1), font_size=16, color=WHITE).next_to(s, DOWN, buff=0.1)
            slots_vgroup.add(VGroup(s, l))
        
        slots_vgroup.arrange(RIGHT, buff=0.15)
        self.place_in_area(slots_vgroup, "C1", "C6", scale_factor=1.0)
        
        # Highlight Parity Slots (1, 2, 4)
        parity_indices = [0, 1, 3] # 1, 2, 4
        for idx in parity_indices:
            slots_vgroup[idx][0].set_stroke(TOWER_COLOR, width=6)
            
        tower_label = Text("Control Towers", font_size=22, color=TOWER_COLOR)
        self.place_in_area(tower_label, 'B1', 'B2', scale_factor=0.8)
        self.place_at_grid(tower_icon, "A1", scale_factor=0.5)

        self.play(Create(slots_vgroup))
        self.play(FadeIn(tower_icon), Write(tower_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(CARGO_COLOR)
        
        # Highlight Cargo Slots (3, 5, 6, 7)
        cargo_indices = [2, 4, 5, 6] # 3, 5, 6, 7
        for idx in cargo_indices:
            slots_vgroup[idx][0].set_stroke(CARGO_COLOR, width=6)
            
        cargo_label = Text("Cargo (Data)", font_size=22, color=CARGO_COLOR)
        self.place_in_area(cargo_label, 'B4', 'B6', scale_factor=0.8)
        self.place_at_grid(cargo_icon, "A5", scale_factor=0.5)
        
        self.play(FadeIn(cargo_icon), Write(cargo_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        # Briefly highlight powers logic
        powers_vgroup = VGroup(
            Text("P1: 1", font_size=18, color=TOWER_COLOR),
            Text("P2: 2", font_size=18, color=TOWER_COLOR),
            Text("P4: 4", font_size=18, color=TOWER_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(powers_vgroup, "D1", scale_factor=0.8)
        
        self.play(Write(powers_vgroup))
        self.wait(1)
        self.play(FadeOut(powers_vgroup))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(TOWER_COLOR)
        
        # Binary address '111' above slot 7
        binary_7 = Text("111", font_size=32, color=WHITE)
        # Resolved Issue 43: Positioned at D6
        self.place_at_grid(binary_7, 'D6', scale_factor=0.8)
        
        # Connections from Slot 7 (index 6) to Towers 1, 2, 4 (indices 0, 1, 3)
        conn_7 = VGroup(
            ArcBetweenPoints(slots_vgroup[6][0].get_top(), slots_vgroup[0][0].get_top(), angle=PI/2.5, color=TOWER_COLOR),
            ArcBetweenPoints(slots_vgroup[6][0].get_top(), slots_vgroup[1][0].get_top(), angle=PI/3.5, color=TOWER_COLOR),
            ArcBetweenPoints(slots_vgroup[6][0].get_top(), slots_vgroup[3][0].get_top(), angle=PI/4.5, color=TOWER_COLOR)
        )
        
        self.play(Write(binary_7))
        self.play(Create(conn_7))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(CARGO_COLOR)
        
        # Show Slot 6 (index 5) connecting only to Towers 4 and 2 (indices 3, 1)
        binary_6 = Text("110", font_size=32, color=WHITE)
        # Resolved Issue 43: Positioned at D5
        self.place_at_grid(binary_6, 'D5', scale_factor=0.8)
        
        conn_6 = VGroup(
            ArcBetweenPoints(slots_vgroup[5][0].get_top(), slots_vgroup[1][0].get_top(), angle=PI/3.5, color=TOWER_COLOR),
            ArcBetweenPoints(slots_vgroup[5][0].get_top(), slots_vgroup[3][0].get_top(), angle=PI/4.5, color=TOWER_COLOR)
        )
        
        self.play(FadeOut(conn_7), FadeOut(binary_7))
        self.play(Write(binary_6), Create(conn_6))
        self.wait(2)
