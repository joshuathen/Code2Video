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
        self.setup_layout("The Power of Generalization", [
            "Abstract vector spaces unify diverse fields.",
            "They connect physics, AI, and signal processing.",
            "Generalization simplifies solving complex problems."
        ])
        
        # Load SVG Assets
        physics_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg")
        ai_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        signals_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/antenna.svg")
        microscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg")
        
        for icon in [physics_icon, ai_icon, signals_icon, microscope]:
            icon.set_color(WHITE)
            
        core = Circle(radius=0.7, color=YELLOW, fill_opacity=0.3)
        core_label = Text("Vector Space", font_size=18, color=YELLOW)
        
        # Positions
        self.place_at_grid(physics_icon, "B2", scale_factor=0.5)
        self.place_at_grid(ai_icon, "B5", scale_factor=0.5)
        self.place_at_grid(signals_icon, "E5", scale_factor=0.5)
        self.place_at_grid(core, "D3", scale_factor=0.9)
        core_label.next_to(core, DOWN, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(physics_icon), FadeIn(ai_icon), FadeIn(signals_icon))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Connections start from center of icons, move to core, avoiding column 1
        lines = VGroup(
            Line(physics_icon.get_center(), core.get_center(), color=YELLOW),
            Line(ai_icon.get_center(), core.get_center(), color=YELLOW),
            Line(signals_icon.get_center(), core.get_center(), color=YELLOW)
        )
        self.play(Create(lines), FadeIn(core), FadeIn(core_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        # Microscope focus effect
        self.place_at_grid(microscope, "D3", scale_factor=0.3)
        self.play(FadeIn(microscope), Indicate(core), core.animate.set_color(GOLD).set_fill(opacity=0.6))
        self.wait(2)
