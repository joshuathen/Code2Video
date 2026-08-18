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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Ternary Bridge", [
            "Ternary systems use three digits: zero, one, two.", 
            "Each digit represents a unique branching path state.", 
            "These represent transitions between three logical pegs."
        ])
        
        # Assets
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        peg_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg")
        
        # Mobjects
        c0 = Circle(radius=0.4, color="#FF5733", fill_opacity=0.6)
        c1 = Circle(radius=0.4, color="#33FF57", fill_opacity=0.6)
        c2 = Circle(radius=0.4, color="#3357FF", fill_opacity=0.6)
        
        l0 = Text("0", font_size=24).move_to(c0.get_center())
        l1 = Text("1", font_size=24).move_to(c1.get_center())
        l2 = Text("2", font_size=24).move_to(c2.get_center())
        
        node0 = VGroup(c0, l0)
        node1 = VGroup(c1, l1)
        node2 = VGroup(c2, l2)
        
        node0_label = Text("0", font_size=24, color="#FF5733")
        node1_label = Text("1", font_size=24, color="#33FF57")
        node2_label = Text("2", font_size=24, color="#3357FF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"), run_time=0.5)
        self.place_at_grid(bridge_icon, "A5", scale_factor=0.5)
        self.play(FadeIn(bridge_icon))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"), run_time=0.5)
        
        # Fix: Layout implementation based on critic feedback
        self.place_at_grid(node0, "B2", scale_factor=0.8)
        self.place_at_grid(node1, "C4", scale_factor=0.8)
        self.place_at_grid(node2, "D6", scale_factor=0.8)
        
        self.place_at_grid(node0_label, "A2", scale_factor=0.6)
        self.place_at_grid(node1_label, "B4", scale_factor=0.6)
        self.place_at_grid(node2_label, "C6", scale_factor=0.6)
        
        self.play(FadeIn(node0), FadeIn(node1), FadeIn(node2), FadeIn(node0_label), FadeIn(node1_label), FadeIn(node2_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"), run_time=0.5)
        
        transition_arc = CurvedArrow(node0.get_center(), node1.get_center(), angle=-TAU/6)
        self.place_in_area(transition_arc, "B2", "C4", scale_factor=0.8)
        
        self.play(Create(transition_arc))
        
        # Use peg icon
        self.place_at_grid(peg_icon, "E4", scale_factor=0.5)
        self.play(FadeIn(peg_icon))
        
        self.wait(1)
