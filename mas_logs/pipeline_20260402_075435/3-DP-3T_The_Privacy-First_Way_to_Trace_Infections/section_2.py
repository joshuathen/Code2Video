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
        # Title and Lecture lines setup
        title_str = "Prerequisite: The Digital Shredder (Hashing) (0:45)"
        lines_str = [
            "A secret seed generates unique, temporary IDs.",
            "Hashing is like a mathematical one-way blender.",
            "IDs are public, but the seed stays hidden."
        ]
        self.setup_layout(title_str, lines_str)
        
        # === Animation for Lecture Line 1 ===
        # A seed icon (#E67E22) labeled "Secret Seed" enters a blender graphic (#95A5A6).
        self.lecture[0].set_color("#E67E22")
        
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/seed.svg
        seed_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/seed.svg", color="#E67E22")
        seed_label = Text("Secret Seed", font_size=18, color="#E67E22")
        seed_group = VGroup(seed_icon, seed_label).arrange(UP, buff=0.2)
        self.place_at_grid(seed_group, "A4", scale_factor=0.6)
        
        # Simple Blender graphic: A trapezoid and a lid
        blender_body = Polygon(
            [-0.8, 1, 0], [0.8, 1, 0], [0.5, -1, 0], [-0.5, -1, 0],
            color="#95A5A6", fill_opacity=0.3, stroke_width=4
        )
        blender_lid = Line([-0.9, 1, 0], [0.9, 1, 0], color="#95A5A6", stroke_width=8)
        blender = VGroup(blender_body, blender_lid)
        
        # Fix for Issue 31 & 33: Move blender up to B3-D5 to utilize vertical space
        self.place_in_area(blender, "B3", "D5", scale_factor=1.2)
        
        self.play(FadeIn(blender))
        self.play(FadeIn(seed_group))
        self.wait(1)
        
        # Animation: Seed enters the blender
        self.play(
            seed_group.animate.move_to(blender.get_center()).scale(0.5).set_opacity(0),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Three different alphanumeric strings (#1ABC9C) emerge from the blender.
        self.lecture[1].set_color("#1ABC9C")
        
        id1 = Text("0x8F2A", font_size=20, color="#1ABC9C")
        id2 = Text("0x3C1B", font_size=20, color="#1ABC9C")
        id3 = Text("0x7E9D", font_size=20, color="#1ABC9C")
        
        # Starting positions inside the blender bottom area
        ids = VGroup(id1, id2, id3)
        for mob in ids:
            mob.move_to(self.grid["D4"])
            mob.set_opacity(0)
        
        # Animation: IDs "flow" out of the bottom of the blender to row F
        self.play(
            id1.animate.move_to(self.grid["F3"]).set_opacity(1),
            id2.animate.move_to(self.grid["F4"]).set_opacity(1),
            id3.animate.move_to(self.grid["F5"]).set_opacity(1),
            run_time=2,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A red arrow (#E74C3C) pointing back from IDs to Seed is crossed out to show one-way hashing.
        self.lecture[2].set_color("#E74C3C")
        
        # Red arrow from IDs back up to the seed's original position
        reverse_arrow = Arrow(
            start=self.grid["F4"], 
            end=self.grid["A4"], 
            color="#E74C3C", 
            stroke_width=8, 
            buff=0.4
        )
        
        # Big Red Cross (X) over the arrow to show "Impossible"
        cross_line1 = Line(UP+LEFT, DOWN+RIGHT, color="#E74C3C", stroke_width=12).scale(0.6)
        cross_line2 = Line(UP+RIGHT, DOWN+LEFT, color="#E74C3C", stroke_width=12).scale(0.6)
        cross = VGroup(cross_line1, cross_line2)
        
        # Fix for Issue 32: Place red cross at C4 (centered on the arrow path)
        self.place_at_grid(cross, "C4")
        
        self.play(Create(reverse_arrow))
        self.wait(0.5)
        self.play(Create(cross))
        self.wait(2)
