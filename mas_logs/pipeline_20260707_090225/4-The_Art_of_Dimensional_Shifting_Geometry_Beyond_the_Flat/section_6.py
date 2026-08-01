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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with refined lecture lines per prompt
        lines = [
            'A 2D knot seems impossible to untie.',
            'Lifting a loop into 3D releases the constraint.',
            'Solve complex problems by seeking the next dimension.'
        ]
        self.setup_layout("Summary: The 'N+1' Mindset", lines)

        # === Animation for Lecture Line 1 ===
        # Show a tangled yellow loop (#FFD700) representing a knot 
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/knot.svg] constrained to 2D.
        self.lecture[0].set_color(YELLOW)
        
        knot_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/knot.svg"
        tangled_loop = SVGMobject(knot_asset).set_color("#FFD700")
        
        # Issue 43: Use scale_factor=0.7 and area B2-E5 to avoid overcrowding
        self.place_in_area(tangled_loop, "B2", "E5", scale_factor=0.7)
        
        self.play(Create(tangled_loop), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A cross-over point flashes white (#FFFFFF) as one segment lifts 'up' 
        # into the 3rd dimension to clear the other.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Flash at knot center using grid-based positioning
        flash_dot = Dot(color="#FFFFFF")
        self.place_in_area(flash_dot, "B2", "E5")
        
        # Prepare the untied loop asset for line 3
        loop_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/loop.svg"
        final_loop = SVGMobject(loop_asset).set_color("#00FF00")
        self.place_in_area(final_loop, "B2", "E5", scale_factor=0.7)

        self.play(Flash(flash_dot, color="#FFFFFF", line_length=0.3, flash_radius=0.5, num_lines=12))
        # Visual hint of "lifting" into 3D using color and scale feedback
        self.play(tangled_loop.animate.scale(1.1).set_color(WHITE), run_time=0.5, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The knot unties into a green circle (#00FF00) 
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/loop.svg] 
        # with the text 'N+1: A New Perspective' (#FFFFFF).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        perspective_text = Text("N+1: A New Perspective", font_size=24, color="#FFFFFF")
        # Issue 42: Fix position (F2-F5) and scale (0.6) to avoid overlap and clipping
        self.place_in_area(perspective_text, 'F2', 'F5', scale_factor=0.6)

        self.play(
            Transform(tangled_loop, final_loop),
            Write(perspective_text),
            run_time=2
        )
        self.wait(3)
