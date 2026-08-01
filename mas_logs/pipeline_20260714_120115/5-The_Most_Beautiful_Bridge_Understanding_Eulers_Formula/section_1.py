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
        # Data from storyboard
        title_text = "The Grand Collision of Worlds"
        lecture_lines = [
            "Meet e, pi, and i: three separate mathematical worlds.",
            "For centuries, these constants appeared completely unrelated.",
            "But a hidden bridge exists between these distant islands.",
            "Let's explore the signpost where these paths finally meet.",
            "This is Euler’s formula, the most beautiful in math."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        color_e = "#88CA5E"
        color_pi = "#FF9D00"
        color_i = "#58C4DD"
        color_gold = "#FFD700"
        color_white = "#FFFFFF"

        # Assets
        island_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/island.svg"
        signpost_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/signpost.svg"

        # Time tracker for floating effect (L014: Use ValueTracker for updates)
        self.time_tracker = ValueTracker(0)
        self.add(self.time_tracker)
        self.time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        def make_floater(mobj, amp=0.1, freq=2, phase=0):
            # Capture start position as a copy to avoid mutation issues
            start_pos = mobj.get_center().copy()
            mobj.add_updater(lambda m: m.move_to(start_pos + UP * amp * np.sin(self.time_tracker.get_value() * freq + phase)))

        # Pre-creating islands with assets (L009: Scan and load assets)
        # Island E
        island_e_svg = SVGMobject(island_path, color=color_e, fill_opacity=0.6)
        label_e = Text("e", slant=ITALIC, color=color_e)
        island_e = VGroup(island_e_svg, label_e)
        self.place_at_grid(island_e, "B2", scale_factor=0.8)
        
        # Island PI
        island_pi_svg = SVGMobject(island_path, color=color_pi, fill_opacity=0.6)
        label_pi = Text("π", color=color_pi)
        island_pi = VGroup(island_pi_svg, label_pi)
        self.place_at_grid(island_pi, "D5", scale_factor=0.8)
        
        # Island I (Fix Issue 25: Move to B5 to avoid edge)
        island_i_svg = SVGMobject(island_path, color=color_i, fill_opacity=0.6)
        label_i = Text("i", slant=ITALIC, color=color_i)
        island_i = VGroup(island_i_svg, label_i)
        self.place_at_grid(island_i, "B5", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Step 1: Display three floating 'islands' [Asset: island.svg] with 'e', 'pi', and 'i' symbols.
        self.play(
            self.lecture[0].animate.set_color(color_white), # Already white, but keeping structure
            FadeIn(island_e),
            FadeIn(island_pi),
            FadeIn(island_i)
        )
        make_floater(island_e, phase=0)
        make_floater(island_pi, phase=np.pi/3)
        make_floater(island_i, phase=2*np.pi/3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Animate the islands [Asset: island.svg] drifting closer together.
        # Clear updaters before repositioning
        island_e.clear_updaters()
        island_pi.clear_updaters()
        island_i.clear_updaters()

        target_area_center = self.grid["D4"] # Intermediate goal for drift

        self.play(
            self.lecture[1].animate.set_color(color_white),
            island_e.animate.move_to(self.grid["C3"]),
            island_pi.animate.move_to(self.grid["D4"]),
            island_i.animate.move_to(self.grid["C5"]),
            run_time=2
        )
        
        # Restart floating at new positions
        make_floater(island_e, phase=0)
        make_floater(island_pi, phase=np.pi/3)
        make_floater(island_i, phase=2*np.pi/3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step 3: Merge the islands into a single, unified landscape with a golden glow (#FFD700).
        island_e.clear_updaters()
        island_pi.clear_updaters()
        island_i.clear_updaters()

        # Unified landscape (Fix Issue 26: Use C3-E5)
        landscape_svg = SVGMobject(island_path, color=color_gold, fill_opacity=0.8)
        self.place_in_area(landscape_svg, "C3", "E5", scale_factor=2.0)
        glow = landscape_svg.copy().set_stroke(color_gold, width=10).set_fill(opacity=0)
        unified_landscape = VGroup(landscape_svg, glow)

        self.play(
            self.lecture[2].animate.set_color(color_gold),
            island_e.animate.move_to(landscape_svg.get_center()).scale(0.5),
            island_pi.animate.move_to(landscape_svg.get_center()).scale(0.5),
            island_i.animate.move_to(landscape_svg.get_center()).scale(0.5),
            run_time=1.5
        )
        self.play(
            FadeOut(island_e, island_pi, island_i),
            FadeIn(unified_landscape)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Step 4: Place a signpost [Asset: signpost.svg] in the center of the merged landscape.
        signpost = SVGMobject(signpost_path, color=WHITE)
        # Position signpost slightly above the landscape center to make room for the sign part
        self.place_in_area(signpost, "C3", "E5", scale_factor=1.2)
        
        self.play(
            self.lecture[3].animate.set_color(color_white),
            FadeIn(signpost, shift=UP)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Step 5: Fade in the formula e^(i*pi) + 1 = 0 on the signpost [Asset: signpost.svg] in white (#FFFFFF).
        # Fix Issue 27: Use scale_factor=1.0 for readability
        # L015: Area spanning multiple columns but few rows
        formula = MarkupText('<i>e</i><sup><i>i</i>π</sup> + 1 = 0', color=color_white)
        self.place_in_area(formula, "D3", "D5", scale_factor=1.0)
        # Move formula to align with the signpost's board (visual estimation based on SVG typical layout)
        formula.move_to(signpost.get_center() + UP * 0.3)

        self.play(
            self.lecture[4].animate.set_color(color_white),
            FadeIn(formula)
        )
        self.wait(3)
