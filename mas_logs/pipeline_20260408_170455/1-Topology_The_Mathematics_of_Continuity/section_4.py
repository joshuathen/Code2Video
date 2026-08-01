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
        # Setup title and lecture lines
        lecture_lines_text = [
            'We classify shapes using properties that never change, called invariants.',
            'The "genus" counts the number of holes in a surface.',
            'A sphere has genus zero, having no holes at all.',
            'A torus has genus one, like a wedding ring.',
            "You can't turn a sphere into a torus without tearing."
        ]
        self.setup_layout("Topological Invariants: The 'Genus'", lecture_lines_text)
        
        # Colors
        COLOR_SPHERE = "#56B4E9"
        COLOR_TORUS = "#009E73"
        COLOR_DOUBLE_TORUS = "#CC79A7"
        COLOR_HIGHLIGHT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # 'We classify shapes using properties that never change, called invariants.'
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        summary_text = Text("Genus is an Invariant", color=WHITE, font_size=24)
        self.place_in_area(summary_text, 'A1', 'A6', scale_factor=1.0)
        self.play(Write(summary_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 'The "genus" counts the number of holes in a surface.'
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'A sphere has genus zero, having no holes at all.'
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_SPHERE)
        )

        # 3D-looking pseudo-cube
        cube_front = Square(side_length=1.5, color=COLOR_SPHERE, fill_opacity=0.3)
        cube_side = Square(side_length=1.5, color=COLOR_SPHERE, fill_opacity=0.3).apply_matrix([[1, 0, 0], [0.4, 1, 0], [0, 0, 1]]).scale(0.5)
        cube_top = Square(side_length=1.5, color=COLOR_SPHERE, fill_opacity=0.3).apply_matrix([[1, 0.4, 0], [0, 1, 0], [0, 0, 1]]).scale(0.5)
        
        cube_side.move_to(cube_front.get_right() + RIGHT*0.3 + UP*0.3)
        cube_top.move_to(cube_front.get_top() + UP*0.3 + RIGHT*0.3)
        cube = VGroup(cube_front, cube_side, cube_top)
        
        self.place_in_area(cube, 'B1', 'D2', scale_factor=0.8)
        self.play(Create(cube))
        
        # Blue sphere (2D representation)
        sphere = Circle(radius=0.75, color=COLOR_SPHERE, fill_opacity=0.8)
        self.place_in_area(sphere, 'B1', 'D2', scale_factor=1.0)
        
        label_0 = Text("Genus 0", color=COLOR_SPHERE, font_size=20)
        self.place_in_area(label_0, 'E1', 'E2', scale_factor=1.0)

        self.play(ReplacementTransform(cube, sphere), FadeIn(label_0))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 'A torus has genus one, like a wedding ring.'
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_TORUS)
        )

        # Torus using provided ring asset
        torus = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ring.svg")
        torus.set_color(COLOR_TORUS)
        self.place_in_area(torus, 'B3', 'D4', scale_factor=1.0)
        
        label_1 = Text("Genus 1", color=COLOR_TORUS, font_size=20)
        self.place_in_area(label_1, 'E3', 'E4', scale_factor=1.0)

        # Double Torus
        dt_base = RoundedRectangle(height=1.0, width=1.8, corner_radius=0.4, color=COLOR_DOUBLE_TORUS, fill_opacity=0.8)
        h1 = Circle(radius=0.2).shift(LEFT * 0.4)
        h2 = Circle(radius=0.2).shift(RIGHT * 0.4)
        double_torus = Cutout(dt_base, h1, h2, color=COLOR_DOUBLE_TORUS, fill_opacity=0.8)
        self.place_in_area(double_torus, 'B5', 'D6', scale_factor=1.0)
        
        label_2 = Text("Genus 2", color=COLOR_DOUBLE_TORUS, font_size=20)
        self.place_in_area(label_2, 'E5', 'E6', scale_factor=1.0)

        self.play(FadeIn(torus), FadeIn(label_1), FadeIn(double_torus), FadeIn(label_2))
        self.wait(1)

        # Highlight holes with flash
        # Torus flash
        flash_1 = Circle(radius=0.1, stroke_width=0, fill_color=COLOR_HIGHLIGHT, fill_opacity=0.8)
        flash_1.move_to(torus.get_center())
        
        # Double torus flashes
        flash_2a = flash_1.copy().move_to(double_torus.get_center() + LEFT*0.4)
        flash_2b = flash_1.copy().move_to(double_torus.get_center() + RIGHT*0.4)
        
        self.play(
            flash_1.animate.scale(4).set_opacity(0),
            flash_2a.animate.scale(4).set_opacity(0),
            flash_2b.animate.scale(4).set_opacity(0),
            run_time=1.5
        )
        self.remove(flash_1, flash_2a, flash_2b)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "You can't turn a sphere into a torus without tearing."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Making a real cross between sphere (col 1-2) and torus (col 3-4)
        cross = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color=RED),
            Line(UP+RIGHT, DOWN+LEFT, color=RED)
        )
        self.place_in_area(cross, 'C2', 'C3', scale_factor=0.3)
        
        self.play(Create(cross))
        self.wait(2)
