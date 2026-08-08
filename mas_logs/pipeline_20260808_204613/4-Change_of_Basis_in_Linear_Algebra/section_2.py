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
        self.setup_layout("The Problem: Viewing Space from Different Angles", 
                          ["View the same vector from different angles.", 
                           "Basis B and Basis C are perspectives.", 
                           "We need a bridge between these views."])
        
        # Setup Axes/Plane
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'C3', 'F5', scale_factor=0.5)
        self.add(axes)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        basis_b = VGroup(Arrow(ORIGIN, axes.c2p(1, 0), color="#00FFFF"), 
                         Arrow(ORIGIN, axes.c2p(0, 1), color="#00FFFF"))
        basis_c = VGroup(Arrow(ORIGIN, axes.c2p(0.7, 0.7), color="#FF00FF"), 
                         Arrow(ORIGIN, axes.c2p(-0.7, 0.7), color="#FF00FF"))
        
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_at_grid(bridge_icon, 'B5', scale_factor=0.3)
        
        self.add(basis_b, basis_c, bridge_icon)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        point_p = Dot(axes.c2p(1, 1), color=YELLOW)
        self.place_at_grid(point_p, 'D4', scale_factor=0.7)
        label_p = MathTex("P", color=YELLOW).next_to(point_p, UP, buff=0.1)
        self.add(point_p, label_p)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        bridge_label = Text("Bridge", font_size=20, color=WHITE)
        self.place_at_grid(bridge_label, 'E3', scale_factor=0.6)
        self.add(bridge_label)
        self.wait(1)
