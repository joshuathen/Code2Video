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
        # Setup layout
        title_text = "Phase 2: The Local Diary (Storage)"
        lines = [
            "Bob's phone keeps a private log of anonymous IDs.",
            "No GPS data or names are ever stored here.",
            "This local diary stays strictly on his own device."
        ]
        self.setup_layout(title_text, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Phone Asset (Issue 29)
        phone_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg")
        # Issue 49: Positioning phone_frame (now phone_svg) B2-E4
        self.place_in_area(phone_svg, "B2", "E4", scale_factor=2.5)
        phone_svg.set_color(BLUE_B)
        
        # Local Log Header
        log_title = Text("Local Log", font_size=20, color=BLUE_A)
        # Issue 49: Positioning log_title at B3
        self.place_at_grid(log_title, "B3")
        
        # Table content (Random strings)
        row1 = Text("XJ-9", font_size=18, color=WHITE)
        row2 = Text("PL-2", font_size=18, color=WHITE)
        row3 = Text("TR-7", font_size=18, color=WHITE)
        
        # Issue 49: Positioning rows at C3, D3, E3
        self.place_at_grid(row1, "C3")
        self.place_at_grid(row2, "D3")
        self.place_at_grid(row3, "E3")
        
        log_content = VGroup(log_title, row1, row2, row3)
        
        self.play(DrawBorderThenFill(phone_svg))
        self.play(Write(log_content))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Metadata labels that are NOT included
        gps_label = Text("GPS Data", font_size=20, color="#EC7063")
        name_label = Text("Name", font_size=20, color="#EC7063")
        
        # Issue 49: Positioning labels at B5 and C5
        self.place_at_grid(gps_label, "B5")
        self.place_at_grid(name_label, "C5")
        
        # Red X marks
        cross1 = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color="#EC7063"),
            Line(UP+RIGHT, DOWN+LEFT, color="#EC7063")
        ).scale(0.3).move_to(gps_label.get_center())
        
        cross2 = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color="#EC7063"),
            Line(UP+RIGHT, DOWN+LEFT, color="#EC7063")
        ).scale(0.3).move_to(name_label.get_center())
        
        self.play(Create(gps_label), Create(name_label))
        self.play(Create(cross1), Create(cross2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Local Log table glows to show data stays local
        glow_rect = SurroundingRectangle(log_content, color="#58D68D", buff=0.2)
        
        self.play(Create(glow_rect))
        self.play(Indicate(log_content, color="#58D68D", scale_factor=1.1))
        self.play(FadeOut(glow_rect))
        
        self.wait(2)
        
        # Final cleanup/state
        self.lecture[2].set_color(WHITE)
        self.wait(1)
