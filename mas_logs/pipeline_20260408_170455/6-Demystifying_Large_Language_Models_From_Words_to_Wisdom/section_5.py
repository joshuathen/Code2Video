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
        # Setup basic layout
        lines = [
            "Data travels through layers like a factory assembly line.",
            "Each station analyzes grammar, logic, or social context.",
            "These layers refine word positions for better accuracy."
        ]
        self.setup_layout("The Transformer Layer: Pattern Processing", lines)
        
        # Define line colors
        line_colors = [YELLOW_A, TEAL_A, GREEN_A]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(line_colors[0])
        
        # Conveyor belt using Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/conveyor.svg
        belt_rect = Rectangle(height=0.6, width=6.0, fill_color="#333333", fill_opacity=1.0, stroke_color=GRAY)
        self.place_in_area(belt_rect, "D1", "D6")
        
        belt_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/conveyor.svg")
        belt_svg.set_color(GRAY_B).set_width(5.8)
        self.place_in_area(belt_svg, "D1", "D6")
        
        conveyor = VGroup(belt_rect, belt_svg)
        
        # Word block 'Apple'
        apple_box = Rectangle(height=0.5, width=0.8, fill_color=WHITE, fill_opacity=0.2, stroke_color=WHITE)
        apple_text = Text("Apple", font_size=18, color=WHITE)
        apple_block = VGroup(apple_box, apple_text)
        self.place_at_grid(apple_block, "D1")
        
        self.play(FadeIn(conveyor), FadeIn(apple_block))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(line_colors[1])
        
        # Grammar Goggles Station (Blue) with Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/g.svg
        grammar_box = Rectangle(height=1.0, width=0.8, fill_color="#00AAFF", fill_opacity=0.3, stroke_color="#00AAFF")
        grammar_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/g.svg").set_color("#00AAFF")
        grammar_label = Text("Grammar Goggles", font_size=24, color="#00AAFF")
        
        self.place_at_grid(grammar_box, "D2")
        self.place_at_grid(grammar_icon, "B2", scale_factor=0.5)
        self.place_in_area(grammar_label, 'C1', 'C3', scale_factor=0.6)
        
        # Context Cameras Station (Green)
        context_box = Rectangle(height=1.0, width=0.8, fill_color="#00FF00", fill_opacity=0.3, stroke_color="#00FF00")
        context_label = Text("Context Cameras", font_size=24, color="#00FF00")
        
        self.place_at_grid(context_box, "D4")
        self.place_in_area(context_label, 'C3', 'C5', scale_factor=0.6)

        # Logic Lenses Station (Purple/Logic)
        logic_box = Rectangle(height=1.0, width=0.8, fill_color="#BB88FF", fill_opacity=0.3, stroke_color="#BB88FF")
        logic_label = Text("Logic Lenses", font_size=24, color="#BB88FF")
        
        self.place_at_grid(logic_box, "D6")
        self.place_in_area(logic_label, 'C5', 'C6', scale_factor=0.6)

        self.play(
            FadeIn(VGroup(grammar_box, grammar_icon, grammar_label)),
            FadeIn(VGroup(context_box, context_label)),
            FadeIn(VGroup(logic_box, logic_label))
        )

        # Movement to Grammar Goggles
        self.play(apple_block.animate.move_to(self.grid["D2"]), run_time=1.5)
        self.play(grammar_box.animate.set_fill(WHITE, opacity=0.8), run_time=0.2)
        self.play(grammar_box.animate.set_fill("#00AAFF", opacity=0.3), run_time=0.2)
        self.wait(0.5)

        # Movement to Context Cameras
        self.play(apple_block.animate.move_to(self.grid["D4"]), run_time=1.5)
        # Change Apple color to #AAFFAA per description
        self.play(
            apple_box.animate.set_fill("#AAFFAA", opacity=0.8),
            apple_text.animate.set_color(BLACK),
            run_time=0.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(line_colors[2])
        
        # Movement to Logic Lenses and final refining position
        target_pos = self.grid["D6"]
        refined_pos = target_pos + RIGHT * 0.1
        
        self.play(apple_block.animate.move_to(target_pos), run_time=1.5)
        self.play(apple_block.animate.move_to(refined_pos), run_time=0.5)
        self.wait(2)
