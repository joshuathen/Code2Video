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
        lecture_lines = ["Multi-head attention runs in parallel.", "Capture different relationship types.", "Synthesize into one final summary."]
        self.setup_layout("Application: Multi-Head Attention", lecture_lines)
        self.lecture.set_opacity(0)
        
        # Color constants
        c_grammar = GREEN
        c_sentiment = BLUE
        c_logic = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        
        # Three attention modules with asset
        module1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg", color=c_grammar)
        module2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg", color=c_sentiment)
        module3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg", color=c_logic)
        
        modules = VGroup(module1, module2, module3).arrange(RIGHT, buff=0.5)
        # Addressed issue 32/47: use suggested area and scale
        self.place_in_area(modules, "B2", "D5", scale_factor=0.6)
        self.play(Create(modules))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.lecture[1].set_color(ORANGE)
        
        # Connections
        conn1 = Line(module1.get_bottom(), module1.get_bottom() + DOWN * 0.5, color=c_grammar)
        conn2 = Line(module2.get_bottom(), module2.get_bottom() + DOWN * 0.5, color=c_sentiment)
        conn3 = Line(module3.get_bottom(), module3.get_bottom() + DOWN * 0.5, color=c_logic)
        
        self.play(Create(conn1), Create(conn2), Create(conn3))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        
        # Final output box with asset
        output_box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/processor.svg", color=WHITE)
        # Addressed issue 33/48: use suggested area and scale
        self.place_at_grid(output_box, "E4", scale_factor=0.8)
        
        # Addressed issue 34/49: create and place connecting arrows
        arrows = VGroup(*[Line(m.get_bottom() + DOWN * 0.5, output_box.get_top(), color=WHITE) for m in [module1, module2, module3]])
        
        self.play(Create(output_box), Create(arrows))
        self.wait(2)
